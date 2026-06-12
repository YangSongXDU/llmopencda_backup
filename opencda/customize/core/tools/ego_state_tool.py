# -*- coding: utf-8 -*-
"""Ego state tool for LLM Agent demos."""

from opencda.customize.core.tools.sensor_tool_base import \
    SensorToolBase, ToolResult


class EgoStateTool(SensorToolBase):
    """
    Read the ego vehicle state from OpenCDA VehicleManager.
    """

    def __init__(self, cost=0.1, enabled=True):
        super(EgoStateTool, self).__init__('ego_state_tool', cost, enabled)

    def run(self, context):
        if not self.enabled:
            return self.disabled_result()

        vehicle_manager = context.get('vehicle_manager', None)
        if vehicle_manager is None:
            return ToolResult(
                self.tool_name,
                success=False,
                cost=0.0,
                reason='vehicle_manager missing in context.'
            )

        vehicle = vehicle_manager.vehicle
        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation

        ego_speed = 0.0
        try:
            ego_speed = float(vehicle_manager.localizer.get_ego_spd())
        except Exception:
            pass

        data = {
            'ego_speed': ego_speed,
            'ego_x': float(location.x),
            'ego_y': float(location.y),
            'ego_z': float(location.z),
            'ego_yaw': float(rotation.yaw)
        }
        return ToolResult(self.tool_name, True, data, self.cost,
                          reason='Ego state retrieved.')
