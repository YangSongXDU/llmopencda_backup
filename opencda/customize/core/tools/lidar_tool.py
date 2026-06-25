# -*- coding: utf-8 -*-
"""
LiDAR tool for LLM Agent demos.

This tool estimates the nearest front obstacle distance from ego LiDAR points.
It never reads CARLA server actor locations. The implementation is deliberately
conservative but now supports automatic front-axis checking because different
sensor transforms may make the forward direction appear as +x or -x in the raw
point cloud frame.
"""

import numpy as np

from opencda.customize.core.tools.sensor_tool_base import \
    SensorToolBase, ToolResult


class LiDARTool(SensorToolBase):
    """Estimate front obstacle distance using ego-mounted LiDAR point cloud."""

    def __init__(self, config=None):
        config = config or {}
        super(LiDARTool, self).__init__(
            'lidar_tool',
            cost=config.get('cost', 2.0),
            enabled=config.get('enabled', True))

        self.front_x_min = float(config.get('front_x_min', 5.0))
        self.front_x_max = float(config.get('front_x_max', 80.0))
        self.lane_y_abs = float(config.get('lane_y_abs', 2.5))
        self.z_min = float(config.get('z_min', -2.2))
        self.z_max = float(config.get('z_max', 2.0))
        self.axis_mode = str(config.get('axis_mode', 'auto'))

        self.min_points = int(config.get('min_points', 5))
        self.x_bin_size = float(config.get('x_bin_size', 3.0))
        self.min_points_per_bin = int(config.get('min_points_per_bin', 3))
        self.distance_percentile = float(config.get('distance_percentile', 30.0))

    def _empty_result(self, reason, point_count=0, candidate_count_pos=0,
                      candidate_count_neg=0):
        return ToolResult(
            self.tool_name,
            success=True,
            data={
                'front_obstacle_detected': False,
                'front_obstacle_distance': 999.0,
                'point_count': int(point_count),
                'candidate_count_pos': int(candidate_count_pos),
                'candidate_count_neg': int(candidate_count_neg),
                'selected_bin_point_count': 0,
                'selected_bin_start': -1.0,
                'selected_bin_end': -1.0,
                'front_axis': 'none',
                'confidence': 0.0
            },
            cost=self.cost,
            reason=reason
        )

    def _extract_candidates(self, points, sign):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        axial = sign * x
        mask = (
            (axial > self.front_x_min) &
            (axial < self.front_x_max) &
            (np.abs(y) < self.lane_y_abs) &
            (z > self.z_min) &
            (z < self.z_max)
        )
        return points[mask], axial[mask]

    def _select_obstacle_bin(self, candidates, axial_values):
        if candidates.shape[0] < self.min_points:
            return None, -1.0, -1.0, 0

        current = self.front_x_min
        while current < self.front_x_max:
            nxt = current + self.x_bin_size
            bin_mask = (axial_values >= current) & (axial_values < nxt)
            bin_points = candidates[bin_mask]
            bin_count = int(bin_points.shape[0])
            if bin_count >= self.min_points_per_bin:
                return bin_points, current, nxt, bin_count
            current = nxt
        return None, -1.0, -1.0, 0

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
            return self._empty_result('LiDAR frame is empty.')

        points = np.asarray(points)
        if len(points.shape) != 2 or points.shape[1] < 3:
            return self._empty_result('Invalid LiDAR point array shape.')

        pos_points, pos_axial = self._extract_candidates(points, sign=1.0)
        neg_points, neg_axial = self._extract_candidates(points, sign=-1.0)
        pos_count = int(pos_points.shape[0])
        neg_count = int(neg_points.shape[0])

        candidate_sets = []
        if self.axis_mode in ['auto', '+x', 'pos']:
            candidate_sets.append(('+x', pos_points, pos_axial, pos_count))
        if self.axis_mode in ['auto', '-x', 'neg']:
            candidate_sets.append(('-x', neg_points, neg_axial, neg_count))

        best = None
        for axis_name, cand_points, cand_axial, cand_count in candidate_sets:
            selected, start, end, bin_count = self._select_obstacle_bin(
                cand_points, cand_axial)
            if selected is None:
                continue
            if best is None or start < best['start']:
                best = {
                    'axis': axis_name,
                    'points': selected,
                    'start': start,
                    'end': end,
                    'bin_count': bin_count,
                    'candidate_count': cand_count
                }

        if best is None:
            return self._empty_result(
                'No valid dense front obstacle bin found in LiDAR ROI.',
                point_count=max(pos_count, neg_count),
                candidate_count_pos=pos_count,
                candidate_count_neg=neg_count)

        selected_points = best['points']
        distances = np.sqrt(selected_points[:, 0] ** 2 + selected_points[:, 1] ** 2)
        front_distance = float(np.percentile(distances, self.distance_percentile))
        confidence = min(1.0, best['bin_count'] / float(max(self.min_points_per_bin * 4, 1)))

        return ToolResult(
            self.tool_name,
            success=True,
            data={
                'front_obstacle_detected': True,
                'front_obstacle_distance': front_distance,
                'point_count': int(best['candidate_count']),
                'candidate_count_pos': pos_count,
                'candidate_count_neg': neg_count,
                'selected_bin_point_count': int(best['bin_count']),
                'selected_bin_start': float(best['start']),
                'selected_bin_end': float(best['end']),
                'front_axis': best['axis'],
                'confidence': confidence
            },
            cost=self.cost,
            reason='Front obstacle distance estimated from ego LiDAR ROI with auto front-axis selection.'
        )
