# -*- coding: utf-8 -*-
"""Simple selective fusion tool for LLM Agent demos."""

from opencda.customize.core.tools.sensor_tool_base import \
    SensorToolBase, ToolResult


class FusionTool(SensorToolBase):
    """
    Fuse structured camera and LiDAR summaries.

    The first version trusts the LiDAR distance when it is available.
    It exists to measure when fusion is called, instead of forcing full
    fusion every simulation step.
    """

    def __init__(self, config=None):
        config = config or {}
        super(FusionTool, self).__init__(
            'fusion_tool',
            cost=config.get('cost', 3.0),
            enabled=config.get('enabled', True))

    def run(self, context):
        if not self.enabled:
            return self.disabled_result()

        tool_results = context.get('tool_results', {}) or {}
        lidar = tool_results.get('lidar_tool', {}) or {}
        camera = tool_results.get('camera_tool', {}) or {}

        lidar_detected = bool(lidar.get('front_obstacle_detected', False))
        lidar_distance = float(lidar.get('front_obstacle_distance', 999.0))
        camera_available = bool(camera.get('image_available', False))

        detected = lidar_detected
        confidence = float(lidar.get('confidence', 0.0))
        if camera_available and detected:
            confidence = min(1.0, confidence + 0.05)

        return ToolResult(
            self.tool_name,
            success=True,
            data={
                'front_vehicle_detected': detected,
                'front_vehicle_distance': lidar_distance if detected else 999.0,
                'fusion_mode': 'camera_lidar_summary_fusion',
                'confidence': confidence
            },
            cost=self.cost,
            reason='Fused structured camera availability and LiDAR front-distance summary.'
        )
