# -*- coding: utf-8 -*-
"""
Camera tool for LLM Agent demos.

This version uses only the ego camera image. It provides a lightweight
front-ROI visual objectness cue without using CARLA server actor locations.
The cue is intentionally simple and is designed as a placeholder before
replacing it with YOLO/VLM-based perception.
"""

import numpy as np

from opencda.customize.core.tools.sensor_tool_base import \
    SensorToolBase, ToolResult


class CameraTool(SensorToolBase):
    """
    Read ego camera frame status and compute a lightweight front-ROI cue.
    """

    def __init__(self, config=None):
        config = config or {}
        super(CameraTool, self).__init__(
            'camera_tool',
            cost=config.get('cost', 1.0),
            enabled=config.get('enabled', True))
        self.roi = config.get('front_roi', [0.30, 0.45, 0.70, 0.85])
        self.edge_threshold = float(config.get('edge_threshold', 35.0))
        self.edge_density_threshold = float(config.get(
            'edge_density_threshold', 0.035))
        self.dark_ratio_threshold = float(config.get(
            'dark_ratio_threshold', 0.08))
        self.dark_pixel_threshold = float(config.get(
            'dark_pixel_threshold', 80.0))

    def run(self, context):
        if not self.enabled:
            return self.disabled_result()

        perception_manager = context.get('perception_manager', None)
        if perception_manager is None:
            return ToolResult(
                self.tool_name, False, cost=0.0,
                reason='perception_manager missing in context.')

        cameras = getattr(perception_manager, 'rgb_camera', None)
        if not cameras:
            return ToolResult(
                self.tool_name,
                success=False,
                data={
                    'image_available': False,
                    'possible_front_vehicle': False,
                    'confidence': 0.0
                },
                cost=self.cost,
                reason='Camera is not spawned. Set camera.visualize > 0 or perception.activate: true.'
            )

        camera = cameras[0]
        image = getattr(camera, 'image', None)
        if image is None:
            return ToolResult(
                self.tool_name,
                success=True,
                data={
                    'image_available': False,
                    'possible_front_vehicle': False,
                    'confidence': 0.0,
                    'frame': getattr(camera, 'frame', 0)
                },
                cost=self.cost,
                reason='Camera image is not available yet.'
            )

        h, w = image.shape[0], image.shape[1]
        x1 = int(max(0.0, min(1.0, self.roi[0])) * w)
        y1 = int(max(0.0, min(1.0, self.roi[1])) * h)
        x2 = int(max(0.0, min(1.0, self.roi[2])) * w)
        y2 = int(max(0.0, min(1.0, self.roi[3])) * h)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = int(0.30 * w), int(0.45 * h), int(0.70 * w), int(0.85 * h)

        roi_img = image[y1:y2, x1:x2, :].astype(np.float32)
        gray = np.mean(roi_img, axis=2)
        if gray.size == 0:
            edge_density = 0.0
            dark_ratio = 0.0
            contrast = 0.0
        else:
            gx = np.abs(np.diff(gray, axis=1))
            gy = np.abs(np.diff(gray, axis=0))
            edge_density = float(
                (np.sum(gx > self.edge_threshold) +
                 np.sum(gy > self.edge_threshold)) /
                max(gx.size + gy.size, 1))
            dark_ratio = float(np.mean(gray < self.dark_pixel_threshold))
            contrast = float(np.std(gray))

        possible_front_vehicle = (
            edge_density > self.edge_density_threshold and
            dark_ratio > self.dark_ratio_threshold
        )
        confidence = min(1.0, 0.5 * edge_density / max(self.edge_density_threshold, 1e-6) +
                         0.5 * dark_ratio / max(self.dark_ratio_threshold, 1e-6))
        if not possible_front_vehicle:
            confidence = min(confidence, 0.45)

        data = {
            'image_available': True,
            'image_height': int(h),
            'image_width': int(w),
            'frame': int(getattr(camera, 'frame', 0)),
            'possible_front_vehicle': bool(possible_front_vehicle),
            'confidence': float(confidence),
            'roi_edge_density': edge_density,
            'roi_dark_ratio': dark_ratio,
            'roi_contrast': contrast,
            'front_roi': [x1, y1, x2, y2]
        }
        return ToolResult(
            self.tool_name, True, data, self.cost,
            reason='Front ROI visual objectness estimated from ego camera image only.'
        )
