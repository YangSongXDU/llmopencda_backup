# -*- coding: utf-8 -*-
"""
Lane availability and adjacent-lane safety check tool.

For the first overtaking prototype this tool uses CARLA map and actor states to
check whether the left/right lane exists and whether nearby vehicles make a lane
change unsafe. It should be treated as a high-level debug/planning tool, not a
camera/LiDAR perception model.
"""

import math

import carla

from opencda.customize.core.tools.sensor_tool_base import \
    SensorToolBase, ToolResult


class LaneCheckTool(SensorToolBase):
    """Check whether adjacent lanes exist and are clear enough."""

    def __init__(self, config=None):
        config = config or {}
        super(LaneCheckTool, self).__init__(
            'lane_check_tool',
            cost=config.get('cost', 0.5),
            enabled=config.get('enabled', False))
        self.safe_front_distance = float(config.get('safe_front_distance', 30.0))
        self.safe_rear_distance = float(config.get('safe_rear_distance', 15.0))
        self.check_distance = float(config.get('check_distance', 60.0))

    @staticmethod
    def _longitudinal(ego_transform, target_location):
        ego_loc = ego_transform.location
        yaw = math.radians(ego_transform.rotation.yaw)
        dx = target_location.x - ego_loc.x
        dy = target_location.y - ego_loc.y
        return dx * math.cos(yaw) + dy * math.sin(yaw)

    @staticmethod
    def _lane_exists(lane_wp, ego_wp):
        if lane_wp is None:
            return False
        if lane_wp.lane_type != carla.LaneType.Driving:
            return False
        # Keep the same driving direction side. Opposite-sign lane ids usually
        # indicate opposite traffic direction on bidirectional roads.
        if lane_wp.lane_id * ego_wp.lane_id <= 0:
            return False
        return True

    def _check_lane_clear(self, ego_vehicle, ego_transform, carla_map, lane_wp):
        if lane_wp is None:
            return False, 999.0, 999.0

        front_min = 999.0
        rear_min = 999.0
        actors = ego_vehicle.get_world().get_actors().filter('*vehicle*')
        for actor in actors:
            if actor.id == ego_vehicle.id:
                continue
            loc = actor.get_location()
            actor_wp = carla_map.get_waypoint(loc)
            if actor_wp.road_id != lane_wp.road_id or actor_wp.lane_id != lane_wp.lane_id:
                continue
            longitudinal = self._longitudinal(ego_transform, loc)
            if longitudinal >= 0.0:
                front_min = min(front_min, longitudinal)
            else:
                rear_min = min(rear_min, abs(longitudinal))

        clear = (front_min >= self.safe_front_distance and
                 rear_min >= self.safe_rear_distance)
        return bool(clear), float(front_min), float(rear_min)

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
        carla_map = vehicle_manager.carla_map
        ego_wp = carla_map.get_waypoint(ego_transform.location)
        left_wp = ego_wp.get_left_lane()
        right_wp = ego_wp.get_right_lane()

        left_exists = self._lane_exists(left_wp, ego_wp)
        right_exists = self._lane_exists(right_wp, ego_wp)

        left_clear, left_front, left_rear = self._check_lane_clear(
            ego_vehicle, ego_transform, carla_map, left_wp if left_exists else None)
        right_clear, right_front, right_rear = self._check_lane_clear(
            ego_vehicle, ego_transform, carla_map, right_wp if right_exists else None)

        data = {
            'current_lane_id': int(ego_wp.lane_id),
            'current_road_id': int(ego_wp.road_id),
            'left_lane_exists': bool(left_exists),
            'left_lane_clear': bool(left_clear) if left_exists else False,
            'left_front_gap': left_front,
            'left_rear_gap': left_rear,
            'left_lane_id': int(left_wp.lane_id) if left_exists else 0,
            'right_lane_exists': bool(right_exists),
            'right_lane_clear': bool(right_clear) if right_exists else False,
            'right_front_gap': right_front,
            'right_rear_gap': right_rear,
            'right_lane_id': int(right_wp.lane_id) if right_exists else 0,
            'debug_only': True
        }

        return ToolResult(
            self.tool_name,
            success=True,
            data=data,
            cost=self.cost,
            reason='Adjacent lane existence and clearance checked.'
        )
