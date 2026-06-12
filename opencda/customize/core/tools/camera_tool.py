# -*- coding: utf-8 -*-
"""
Camera tool for LLM Agent demos.

The first version provides a structured camera availability summary.
It does not perform YOLO detection yet. This keeps the tool interface
ready for a later VLM/YOLO upgrade without blocking the LiDAR-first demo.
"""

from opencda.customize.core.tools.sensor_tool_base import \
    SensorToolBase, ToolResult


class CameraTool(SensorToolBase):
    """
    Read ego camera frame status and return a lightweight summary.
    """

    def __init__(self, config=None):
        config = config or {}
        super(CameraTool, self).__init__(
            'camera_tool',
            cost=config.get('cost', 1.0),
            enabled=config.get('enabled', True))

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

        data = {
            'image_available': True,
            'image_height': int(image.shape[0]),
            'image_width': int(image.shape[1]),
            'frame': int(getattr(camera, 'frame', 0)),
            'possible_front_vehicle': False,
            'confidence': 0.5
        }
        return ToolResult(self.tool_name, True, data, self.cost,
                          reason='Camera frame is available. Detection is reserved for future YOLO/VLM upgrade.')
