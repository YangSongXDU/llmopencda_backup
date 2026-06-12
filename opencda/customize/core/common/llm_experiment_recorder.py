# -*- coding: utf-8 -*-
"""CSV recorder for LLM Sensor Tool Agent demos."""

import csv
import os


def _safe_getattr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


class LLMExperimentRecorder(object):
    """Record tool calls, LLM decisions, and control commands."""

    def __init__(self, save_path):
        self.save_path = save_path
        self.rows = []
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.fieldnames = [
            'step',
            'ego_x', 'ego_y', 'ego_z', 'ego_yaw', 'ego_speed',
            'called_tools', 'tool_call_count', 'tool_total_cost',
            'camera_called', 'lidar_called', 'fusion_called',
            'lidar_front_distance', 'lidar_point_count',
            'fusion_front_distance', 'final_front_distance',
            'risk_level', 'driving_advice', 'target_speed_advice',
            'target_speed', 'throttle', 'brake', 'steer',
            'reason'
        ]

    def record_step(self, step, vehicle_manager, control):
        vehicle = vehicle_manager.vehicle
        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation

        try:
            ego_speed = float(vehicle_manager.localizer.get_ego_spd())
        except Exception:
            ego_speed = 0.0

        agent = vehicle_manager.agent
        llm_agent = _safe_getattr(agent, 'llm_sensor_agent', None)
        decision = _safe_getattr(agent, 'last_llm_decision', None)
        tool_results = _safe_getattr(llm_agent, 'last_tool_results', {}) or {}
        called_tools = _safe_getattr(llm_agent, 'last_called_tools', []) or []

        lidar = tool_results.get('lidar_tool', {}) or {}
        fusion = tool_results.get('fusion_tool', {}) or {}

        row = {
            'step': step,
            'ego_x': location.x,
            'ego_y': location.y,
            'ego_z': location.z,
            'ego_yaw': rotation.yaw,
            'ego_speed': ego_speed,
            'called_tools': '|'.join(called_tools),
            'tool_call_count': len(called_tools),
            'tool_total_cost': _safe_getattr(llm_agent, 'last_total_cost', 0.0),
            'camera_called': 'camera_tool' in called_tools,
            'lidar_called': 'lidar_tool' in called_tools,
            'fusion_called': 'fusion_tool' in called_tools,
            'lidar_front_distance': lidar.get('front_obstacle_distance', 999.0),
            'lidar_point_count': lidar.get('point_count', 0),
            'fusion_front_distance': fusion.get('front_vehicle_distance', 999.0),
            'final_front_distance': decision.front_vehicle_distance if decision else 999.0,
            'risk_level': decision.risk_level if decision else '',
            'driving_advice': decision.driving_advice if decision else '',
            'target_speed_advice': decision.target_speed_advice if decision else '',
            'target_speed': _safe_getattr(agent, 'last_target_speed', ''),
            'throttle': control.throttle,
            'brake': control.brake,
            'steer': control.steer,
            'reason': decision.reason if decision else ''
        }
        self.rows.append(row)

    def save(self):
        with open(self.save_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        print('[LLMExperimentRecorder] CSV saved to: %s' % self.save_path)
