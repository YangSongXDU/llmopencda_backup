# -*- coding: utf-8 -*-
"""
Radar tool for LLM Agent demos.

This tool uses ego-mounted CARLA radar data only. It does not read CARLA
server actor locations. The radar sensor is spawned lazily when the tool is
first called, then its asynchronous detections are converted into a structured
front-object summary for the LLM Agent.
"""

import math
import weakref

import carla
import numpy as np

from opencda.customize.core.tools.sensor_tool_base import \
    SensorToolBase, ToolResult


class RadarSensor(object):
    """A minimal CARLA radar sensor wrapper."""

    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.data = []
        self.frame = 0
        self.timestamp = None

        blueprint = self.world.get_blueprint_library().find(
            'sensor.other.radar')
        blueprint.set_attribute('horizontal_fov', str(config.get(
            'horizontal_fov', 24.0)))
        blueprint.set_attribute('vertical_fov', str(config.get(
            'vertical_fov', 8.0)))
        blueprint.set_attribute('points_per_second', str(config.get(
            'points_per_second', 1500)))
        blueprint.set_attribute('range', str(config.get('range', 80.0)))

        position = config.get('position', [2.5, 0.0, 1.0, 0.0])
        spawn_point = carla.Transform(
            carla.Location(x=float(position[0]),
                           y=float(position[1]),
                           z=float(position[2])),
            carla.Rotation(yaw=float(position[3])))
        self.sensor = self.world.spawn_actor(blueprint, spawn_point,
                                             attach_to=vehicle)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: RadarSensor._on_radar_event(
            weak_self, event))

    @staticmethod
    def _on_radar_event(weak_self, event):
        self = weak_self()
        if not self:
            return

        detections = []
        for detect in event:
            depth = float(detect.depth)
            azimuth = float(detect.azimuth)
            altitude = float(detect.altitude)
            velocity = float(detect.velocity)
            x = depth * math.cos(altitude) * math.cos(azimuth)
            y = depth * math.cos(altitude) * math.sin(azimuth)
            z = depth * math.sin(altitude)
            detections.append({
                'depth': depth,
                'azimuth': azimuth,
                'altitude': altitude,
                'velocity': velocity,
                'x': x,
                'y': y,
                'z': z
            })

        self.data = detections
        self.frame = event.frame
        self.timestamp = event.timestamp

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


class RadarTool(SensorToolBase):
    """Detect front moving objects and estimate distance / TTC from radar."""

    def __init__(self, config=None):
        config = config or {}
        super(RadarTool, self).__init__(
            'radar_tool',
            cost=config.get('cost', 1.5),
            enabled=config.get('enabled', True))
        self.config = config
        self.sensor = None
        self.front_x_min = float(config.get('front_x_min', 15.0))
        self.front_x_max = float(config.get('front_x_max', 80.0))
        self.lane_y_abs = float(config.get('lane_y_abs', 1.8))
        self.z_abs = float(config.get('z_abs', 2.5))
        self.min_points = int(config.get('min_points', 3))
        self.depth_bin_size = float(config.get('depth_bin_size', 4.0))
        self.min_points_per_bin = int(config.get('min_points_per_bin', 3))
        self.ttc_max = float(config.get('ttc_max', 99.0))
        self.max_abs_velocity = float(config.get('max_abs_velocity', 80.0))
        self.min_confidence = float(config.get('min_confidence', 0.35))

    def _ensure_sensor(self, vehicle_manager):
        if self.sensor is None:
            self.sensor = RadarSensor(vehicle_manager.vehicle, self.config)

    def _empty_result(self, reason, raw_count=0, roi_count=0, valid_count=0):
        return ToolResult(
            self.tool_name,
            success=True,
            data={
                'front_object_detected': False,
                'front_object_distance': 999.0,
                'front_object_relative_velocity': 0.0,
                'closing_speed': 0.0,
                'ttc': self.ttc_max,
                'radar_point_count': int(valid_count),
                'radar_roi_point_count': int(roi_count),
                'radar_raw_point_count': int(raw_count),
                'selected_bin_point_count': 0,
                'confidence': 0.0,
                'frame': getattr(self.sensor, 'frame', 0) if self.sensor else 0
            },
            cost=self.cost,
            reason=reason
        )

    def run(self, context):
        if not self.enabled:
            return self.disabled_result()

        vehicle_manager = context.get('vehicle_manager', None)
        if vehicle_manager is None:
            return ToolResult(
                self.tool_name, False, cost=0.0,
                reason='vehicle_manager missing in context.')

        self._ensure_sensor(vehicle_manager)
        detections = self.sensor.data or []
        raw_count = len(detections)
        if raw_count == 0:
            return self._empty_result('Radar frame has no detections yet.')

        roi_front = []
        valid_front = []
        for det in detections:
            if (det['x'] > self.front_x_min and
                    det['x'] < self.front_x_max and
                    abs(det['y']) < self.lane_y_abs and
                    abs(det['z']) < self.z_abs):
                roi_front.append(det)
                if abs(det['velocity']) <= self.max_abs_velocity:
                    valid_front.append(det)

        if len(valid_front) < self.min_points:
            return self._empty_result(
                'No stable radar cluster in the ego-lane front ROI.',
                raw_count=raw_count,
                roi_count=len(roi_front),
                valid_count=len(valid_front))

        valid_front = sorted(valid_front, key=lambda d: d['depth'])
        selected = None
        current = self.front_x_min
        while current < self.front_x_max:
            nxt = current + self.depth_bin_size
            bin_points = [d for d in valid_front if current <= d['depth'] < nxt]
            if len(bin_points) >= self.min_points_per_bin:
                selected = bin_points
                break
            current = nxt

        if not selected:
            return self._empty_result(
                'No dense radar depth bin found after clutter filtering.',
                raw_count=raw_count,
                roi_count=len(roi_front),
                valid_count=len(valid_front))

        distances = np.array([d['depth'] for d in selected], dtype=np.float32)
        velocities = np.array([d['velocity'] for d in selected], dtype=np.float32)
        distance = float(np.median(distances))
        rel_velocity = float(np.median(velocities))
        closing_speed = max(0.0, -rel_velocity)
        ttc = distance / closing_speed if closing_speed > 0.1 else self.ttc_max
        ttc = min(float(ttc), self.ttc_max)
        confidence = min(1.0, len(selected) / float(max(self.min_points_per_bin * 3, 1)))

        if confidence < self.min_confidence:
            return self._empty_result(
                'Radar cluster confidence is below threshold.',
                raw_count=raw_count,
                roi_count=len(roi_front),
                valid_count=len(valid_front))

        return ToolResult(
            self.tool_name,
            success=True,
            data={
                'front_object_detected': True,
                'front_object_distance': distance,
                'front_object_relative_velocity': rel_velocity,
                'closing_speed': closing_speed,
                'ttc': ttc,
                'radar_point_count': len(valid_front),
                'radar_roi_point_count': len(roi_front),
                'radar_raw_point_count': raw_count,
                'selected_bin_point_count': len(selected),
                'confidence': confidence,
                'frame': getattr(self.sensor, 'frame', 0)
            },
            cost=self.cost,
            reason='Front object distance and TTC estimated from a filtered radar depth cluster.'
        )

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None
