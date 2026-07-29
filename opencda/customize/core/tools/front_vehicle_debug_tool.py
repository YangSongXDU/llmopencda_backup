# -*- coding: utf-8 -*-
"""
Debug front-vehicle tool for the LLM overtaking demo.

This tool intentionally reads CARLA actor states to provide a ground-truth
front-vehicle reference for debugging, evaluation, and a first overtaking
prototype. It is NOT a pure ego-sensor perception tool. Keep it disabled when
running strict self-perception experiments.
"""

import math

from opencda.core.common.misc import get_speed
from opencda.customize.core.tools.sensor_tool_base import \
    SensorToolBase, ToolResult


class FrontVehicleDebugTool(SensorToolBase):
    """
    Find the nearest same-lane vehicle in front of ego using CARLA actors.

    The result is useful to answer whether distance=999 means "no true front
    car" or simply "the ego sensor tools failed to detect the front car".
    """

    def __init__(self, config=None):
        config = config or {}
        super(FrontVehicleDebugTool, self).__init__(
            'front_vehicle_debug_tool',
            cost=config.get('cost', 0.2),
            enabled=config.get('enabled', False))
        self.max_distance = float(config.get('max_distance', 120.0))
        self.lane_y_abs = float(config.get('lane_y_abs', 2.5))
        self.same_lane_only = bool(config.get('same_lane_only', True))
        self.allow_cross_road_same_lane = bool(
            config.get('allow_cross_road_same_lane', False))

    @staticmethod
    def _longitudinal_lateral(ego_transform, target_location):
        ego_loc = ego_transform.location
        yaw = math.radians(ego_transform.rotation.yaw)
        dx = target_location.x - ego_loc.x
        dy = target_location.y - ego_loc.y
        longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return longitudinal, lateral

    def run(self, context):
        if not self.enabled:
            return self.disabled_result()

        vehicle_manager = context.get('vehicle_manager', None)
        if vehicle_manager is None:
            return ToolResult(
                self.tool_name, False, cost=0.0,
                reason='vehicle_manager missing in context.')

        ego_vehicle = vehicle_manager.vehicle
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        carla_map = vehicle_manager.carla_map
        ego_wp = carla_map.get_waypoint(ego_location)
        ego_speed = 0.0
        try:
            ego_speed = float(vehicle_manager.localizer.get_ego_spd())
        except Exception:
            pass

        actors = ego_vehicle.get_world().get_actors().filter('*vehicle*')
        best = None
        for actor in actors:
            if actor.id == ego_vehicle.id:
                continue
            loc = actor.get_location()
            longitudinal, lateral = self._longitudinal_lateral(ego_transform, loc)
            if longitudinal <= 0.0 or longitudinal > self.max_distance:
                continue
            if abs(lateral) > self.lane_y_abs:
                continue

            actor_wp = carla_map.get_waypoint(loc)
            same_lane = actor_wp.lane_id == ego_wp.lane_id and (
                actor_wp.road_id == ego_wp.road_id or
                self.allow_cross_road_same_lane)
            if self.same_lane_only and not same_lane:
                continue

            if best is None or longitudinal < best['distance']:
                target_speed = float(get_speed(actor))
                best = {
                    'actor_id': actor.id,
                    'distance': float(longitudinal),
                    'lateral_offset': float(lateral),
                    'speed': target_speed,
                    'relative_speed': float(ego_speed - target_speed),
                    'same_lane': bool(same_lane),
                    'lane_id': int(actor_wp.lane_id),
                    'road_id': int(actor_wp.road_id)
                }

        if best is None:
            return ToolResult(
                self.tool_name,
                success=True,
                data={
                    'front_vehicle_detected': False,
                    'front_vehicle_distance': 999.0,
                    'front_vehicle_speed': 0.0,
                    'relative_speed': 0.0,
                    'same_lane': False,
                    'actor_id': -1,
                    'debug_only': True
                },
                cost=self.cost,
                reason='No same-lane front vehicle found by debug ground truth.'
            )

        return ToolResult(
            self.tool_name,
            success=True,
            data={
                'front_vehicle_detected': True,
                'front_vehicle_distance': best['distance'],
                'front_vehicle_speed': best['speed'],
                'relative_speed': best['relative_speed'],
                'front_vehicle_lateral_offset': best['lateral_offset'],
                'same_lane': best['same_lane'],
                'actor_id': best['actor_id'],
                'lane_id': best['lane_id'],
                'road_id': best['road_id'],
                'debug_only': True
            },
            cost=self.cost,
            reason='Same-lane front vehicle found using debug ground truth.'
        )
