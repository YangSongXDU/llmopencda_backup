# -*- coding: utf-8 -*-
"""Selective fusion tool for LLM Agent demos."""

from opencda.customize.core.tools.sensor_tool_base import \
    SensorToolBase, ToolResult


class FusionTool(SensorToolBase):
    """
    Fuse structured camera, LiDAR, and radar summaries.

    This is result-level fusion: all inputs are outputs of ego-mounted
    sensing tools. No CARLA server actor location is used here.
    """

    def __init__(self, config=None):
        config = config or {}
        super(FusionTool, self).__init__(
            'fusion_tool',
            cost=config.get('cost', 3.0),
            enabled=config.get('enabled', True))
        self.max_lidar_radar_distance_gap = float(config.get(
            'max_lidar_radar_distance_gap', 8.0))

    def run(self, context):
        if not self.enabled:
            return self.disabled_result()

        tool_results = context.get('tool_results', {}) or {}
        lidar = tool_results.get('lidar_tool', {}) or {}
        camera = tool_results.get('camera_tool', {}) or {}
        radar = tool_results.get('radar_tool', {}) or {}

        lidar_detected = bool(lidar.get('front_obstacle_detected', False))
        lidar_distance = float(lidar.get('front_obstacle_distance', 999.0))
        lidar_conf = float(lidar.get('confidence', 0.0))

        radar_detected = bool(radar.get('front_object_detected', False))
        radar_distance = float(radar.get('front_object_distance', 999.0))
        radar_conf = float(radar.get('confidence', 0.0))
        radar_velocity = float(radar.get('front_object_relative_velocity', 0.0))
        radar_ttc = float(radar.get('ttc', 99.0))

        camera_available = bool(camera.get('image_available', False))
        camera_possible = bool(camera.get('possible_front_vehicle', False))
        camera_conf = float(camera.get('confidence', 0.0))

        distance_candidates = []
        if lidar_detected:
            distance_candidates.append(('lidar', lidar_distance, max(lidar_conf, 0.01)))
        if radar_detected:
            distance_candidates.append(('radar', radar_distance, max(radar_conf, 0.01)))

        detected = len(distance_candidates) > 0
        if detected:
            weight_sum = sum(item[2] for item in distance_candidates)
            fused_distance = sum(item[1] * item[2] for item in distance_candidates) / weight_sum
        else:
            fused_distance = 999.0

        confidence = max(lidar_conf if lidar_detected else 0.0,
                         radar_conf if radar_detected else 0.0)

        # Cross-modal confirmation bonus.
        if lidar_detected and radar_detected:
            gap = abs(lidar_distance - radar_distance)
            if gap <= self.max_lidar_radar_distance_gap:
                confidence = min(1.0, confidence + 0.15)
            else:
                confidence = max(0.0, confidence - 0.15)
        else:
            gap = 999.0

        if camera_available and camera_possible and detected:
            confidence = min(1.0, confidence + 0.05 * max(camera_conf, 0.1))

        return ToolResult(
            self.tool_name,
            success=True,
            data={
                'front_vehicle_detected': detected,
                'front_vehicle_distance': fused_distance,
                'front_vehicle_relative_velocity': radar_velocity if radar_detected else 0.0,
                'ttc': radar_ttc if radar_detected else 99.0,
                'fusion_mode': 'camera_lidar_radar_summary_fusion',
                'confidence': confidence,
                'lidar_used': lidar_detected,
                'radar_used': radar_detected,
                'camera_confirmed': camera_available and camera_possible,
                'lidar_radar_distance_gap': gap
            },
            cost=self.cost,
            reason='Fused ego-camera, ego-LiDAR and ego-radar structured summaries.'
        )
