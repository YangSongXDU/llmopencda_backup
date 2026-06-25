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
            'horizontal_fov', 30.0)))
        blueprint.set_attribute('vertical_fov', str(config.get(
            'vertical_fov', 10.0)))
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
    """
    Detect front moving objects and estimate distance / TTC from radar.
    """

    def __init__(self, config=None):
        config = config or {}
        super(RadarTool, self).__init__(
            'radar_tool',
            cost=config.get('cost', 1.5),
            enabled=config.get('enabled', True))
        self.config = config
        self.sensor = None
        self.front_x_min = float(config.get('front_x_min', 3.0))
        self.front_x_max = float(config.get('front_x_max', 80.0))
        self.lane_y_abs = float(config.get('lane_y_abs', 2.2))
        self.z_abs = float(config.get('z_abs', 3.0))
        self.min_points = int(config.get('min_points', 1))
        self.ttc_max = float(config.get('ttc_max', 99.0))

    def _ensure_sensor(self, vehicle_manager):
        if self.sensor is None:
            self.sensor = RadarSensor(vehicle_manager.vehicle, self.config)

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
        if len(detections) == 0:
            return ToolResult(
                self.tool_name,
                success=True,
                data={
                    'front_object_detected': False,
                    'front_object_distance': 999.0,
                    'front_object_relative_velocity': 0.0,
                    'ttc': self.ttc_max,
                    'radar_point_count': 0,
                    'confidence': 0.0,
                    'frame': getattr(self.sensor, 'frame', 0)
                },
                cost=self.cost,
                reason='Radar frame has no detections yet.'
            )

        front = []
        for det in detections:
            if (det['x'] > self.front_x_min and
                    det['x'] < self.front_x_max and
                    abs(det['y']) < self.lane_y_abs and
                    abs(det['z']) < self.z_abs):
                front.append(det)

        if len(front) < self.min_points:
            return ToolResult(
                self.tool_name,
                success=True,
                data={
                    'front_object_detected': False,
                    'front_object_distance': 999.0,
                    'front_object_relative_velocity': 0.0,
                    'ttc': self.ttc_max,
                    'radar_point_count': len(front),
                    'confidence': 0.0,
                    'frame': getattr(self.sensor, 'frame', 0)
                },
                cost=self.cost,
                reason='No radar detections in the ego-lane front ROI.'
            )

        # Use the nearest valid radar detection in front.
        front = sorted(front, key=lambda d: d['depth'])
        nearest = front[0]
        distance = float(nearest['depth'])
        rel_velocity = float(nearest['velocity'])

        # CARLA radar velocity is the relative radial velocity. In practice,
        # negative values often represent closing objects in front. We also
        # expose the raw relative velocity for later calibration.
        closing_speed = max(0.0, -rel_velocity)
        ttc = distance / closing_speed if closing_speed > 0.1 else self.ttc_max
        ttc = min(float(ttc), self.ttc_max)
        confidence = min(1.0, len(front) / 5.0)

        return ToolResult(
            self.tool_name,
            success=True,
            data={
                'front_object_detected': True,
                'front_object_distance': distance,
                'front_object_relative_velocity': rel_velocity,
                'closing_speed': closing_speed,
                'ttc': ttc,
                'radar_point_count': len(front),
                'confidence': confidence,
                'frame': getattr(self.sensor, 'frame', 0)
            },
            cost=self.cost,
            reason='Front object distance and TTC estimated from radar detections.'
        )

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None
