# -*- coding: utf-8 -*-
"""
LiDAR tool for LLM Agent demos.

The first version estimates the nearest front obstacle distance from raw
LiDAR points. It does not use CARLA server vehicle positions for control.

This version adds a more conservative front-region filter and x-bin based
selection to avoid treating very near ground/ego-body points as obstacles.
"""

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

        # Front ROI. Keep front_x_min large enough to ignore ego-body/near-field
        # points, and keep z_min above the road plane to avoid ground returns.
        self.front_x_min = float(config.get('front_x_min', 8.0))
        self.front_x_max = float(config.get('front_x_max', 80.0))
        self.lane_y_abs = float(config.get('lane_y_abs', 1.8))
        self.z_min = float(config.get('z_min', -1.2))
        self.z_max = float(config.get('z_max', 1.5))

        # Robust obstacle extraction.
        self.min_points = int(config.get('min_points', 20))
        self.x_bin_size = float(config.get('x_bin_size', 2.0))
        self.min_points_per_bin = int(config.get('min_points_per_bin', 8))
        self.distance_percentile = float(config.get('distance_percentile', 20.0))

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

        # Step 1: remove ego-body, near-field and ground points.
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
                    'confidence': 0.0,
                    'selected_bin_start': -1.0,
                    'selected_bin_end': -1.0
                },
                cost=self.cost,
                reason='No enough front LiDAR points after ROI filtering.'
            )

        # Step 2: scan x bins from near to far. A valid obstacle should occupy
        # enough points in a local forward-distance bin instead of being a
        # single noisy point.
        selected_points = None
        selected_bin_start = -1.0
        selected_bin_end = -1.0
        current = self.front_x_min
        while current < self.front_x_max:
            nxt = current + self.x_bin_size
            bin_mask = (front_points[:, 0] >= current) & (front_points[:, 0] < nxt)
            bin_points = front_points[bin_mask]
            if int(bin_points.shape[0]) >= self.min_points_per_bin:
                selected_points = bin_points
                selected_bin_start = current
                selected_bin_end = nxt
                break
            current = nxt

        if selected_points is None:
            return ToolResult(
                self.tool_name,
                success=True,
                data={
                    'front_obstacle_detected': False,
                    'front_obstacle_distance': 999.0,
                    'point_count': point_count,
                    'confidence': 0.0,
                    'selected_bin_start': -1.0,
                    'selected_bin_end': -1.0
                },
                cost=self.cost,
                reason='No valid dense front obstacle bin found.'
            )

        distances = np.sqrt(selected_points[:, 0] ** 2 + selected_points[:, 1] ** 2)
        front_distance = float(np.percentile(distances, self.distance_percentile))
        confidence = min(1.0, selected_points.shape[0] / float(max(self.min_points_per_bin * 4, 1)))

        return ToolResult(
            self.tool_name,
            success=True,
            data={
                'front_obstacle_detected': True,
                'front_obstacle_distance': front_distance,
                'point_count': point_count,
                'selected_bin_point_count': int(selected_points.shape[0]),
                'selected_bin_start': selected_bin_start,
                'selected_bin_end': selected_bin_end,
                'confidence': confidence
            },
            cost=self.cost,
            reason='Front obstacle distance estimated from robust LiDAR ROI and x-bin filtering.'
        )
