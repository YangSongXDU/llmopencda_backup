# -*- coding: utf-8 -*-
"""
LiDAR tool for LLM Agent demos.

The first version estimates the nearest front obstacle distance from raw
LiDAR points. It does not use CARLA server vehicle positions for control.
"""

import math
import numpy as np

from opencda.customize.core.tools.sensor_tool_base import \
    SensorToolBase, ToolResult


class LiDARTool(SensorToolBase):
    """
    Estimate front obstacle distance using ego-mounted LiDAR point cloud.

    The point cloud coordinate convention follows the ego sensor frame.
    Points in front of the ego vehicle usually have positive x values.
    """

    def __init__(self, config=None):
        config = config or {}
        super(LiDARTool, self).__init__(
            'lidar_tool',
            cost=config.get('cost', 2.0),
            enabled=config.get('enabled', True))
        self.front_x_min = float(config.get('front_x_min', 2.0))
        self.front_x_max = float(config.get('front_x_max', 60.0))
        self.lane_y_abs = float(config.get('lane_y_abs', 2.0))
        self.z_min = float(config.get('z_min', -2.0))
        self.z_max = float(config.get('z_max', 3.0))
        self.min_points = int(config.get('min_points', 5))

    def run(self, context):
        if not self.enabled:
            return self.disabled_result()

        perception_manager = context.get('perception_manager', None)
        if perception_manager is None:
            return ToolResult(
                self.tool_name, False, cost=0.0,
                reason='perception_manager missing in context.')

        lidar = getattr(perception_manager, 'lidar', None)
        if lidar is None or getattr(lidar, 'data', None) is None:
            return ToolResult(
                self.tool_name,
                success=False,
                data={
                    'front_obstacle_detected': False,
                    'front_obstacle_distance': 999.0,
                    'point_count': 0,
                    'confidence': 0.0
                },
                cost=self.cost,
                reason='LiDAR data not available. Set lidar.visualize: true or perception.activate: true.'
            )

        points = lidar.data
        if points is None or len(points) == 0:
            return ToolResult(
                self.tool_name,
                success=True,
                data={
                    'front_obstacle_detected': False,
                    'front_obstacle_distance': 999.0,
                    'point_count': 0,
                    'confidence': 0.0
                },
                cost=self.cost,
                reason='LiDAR frame is empty.'
            )

        points = np.asarray(points)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        mask = (
            (x > self.front_x_min) &
            (x < self.front_x_max) &
            (np.abs(y) < self.lane_y_abs) &
            (z > self.z_min) &
            (z < self.z_max)
        )
        front_points = points[mask]
        point_count = int(front_points.shape[0])

        if point_count < self.min_points:
            return ToolResult(
                self.tool_name,
                success=True,
                data={
                    'front_obstacle_detected': False,
                    'front_obstacle_distance': 999.0,
                    'point_count': point_count,
                    'confidence': 0.0
                },
                cost=self.cost,
                reason='No enough front LiDAR points in the target region.'
            )

        distances = np.sqrt(front_points[:, 0] ** 2 + front_points[:, 1] ** 2)
        front_distance = float(np.min(distances))
        confidence = min(1.0, point_count / 200.0)

        return ToolResult(
            self.tool_name,
            success=True,
            data={
                'front_obstacle_detected': True,
                'front_obstacle_distance': front_distance,
                'point_count': point_count,
                'confidence': confidence
            },
            cost=self.cost,
            reason='Front obstacle distance estimated from LiDAR points.'
        )
