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
    """Record tool calls, LLM decisions, maneuvers, and controls."""

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
            'llm_backend', 'llm_fallback_used', 'llm_error',
            'camera_called', 'lidar_called', 'radar_called', 'fusion_called',
            'front_debug_called', 'lane_check_called',
            'camera_possible_front_vehicle', 'camera_confidence',
            'camera_roi_edge_density', 'camera_roi_dark_ratio',
            'lidar_front_distance', 'lidar_point_count',
            'lidar_candidate_count_pos', 'lidar_candidate_count_neg',
            'lidar_selected_bin_point_count', 'lidar_front_axis',
            'radar_front_distance', 'radar_relative_velocity',
            'radar_ttc', 'radar_point_count', 'radar_roi_point_count',
            'radar_raw_point_count', 'radar_selected_bin_point_count',
            'debug_front_vehicle_detected', 'debug_front_vehicle_distance',
            'debug_front_vehicle_speed', 'debug_front_vehicle_relative_speed',
            'left_lane_exists', 'left_lane_clear', 'left_front_gap', 'left_rear_gap',
            'right_lane_exists', 'right_lane_clear', 'right_front_gap', 'right_rear_gap',
            'fusion_front_distance', 'fusion_ttc', 'fusion_confidence',
            'final_front_distance',
            'risk_level', 'driving_advice', 'target_speed_advice',
            'maneuver', 'target_lane', 'lane_change_required',
            'overtake_state', 'original_lane_id', 'current_lane_id',
            'target_lane_id', 'tracked_front_actor_id',
            'passed_front_vehicle', 'return_lane_clear',
            'maneuver_applied', 'maneuver_reason',
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

        camera = tool_results.get('camera_tool', {}) or {}
        lidar = tool_results.get('lidar_tool', {}) or {}
        radar = tool_results.get('radar_tool', {}) or {}
        front_debug = tool_results.get('front_vehicle_debug_tool', {}) or {}
        lane = tool_results.get('lane_check_tool', {}) or {}
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
            'llm_backend': _safe_getattr(llm_agent, 'last_llm_backend', ''),
            'llm_fallback_used': _safe_getattr(llm_agent, 'last_llm_fallback_used', False),
            'llm_error': _safe_getattr(llm_agent, 'last_llm_error', ''),
            'camera_called': 'camera_tool' in called_tools,
            'lidar_called': 'lidar_tool' in called_tools,
            'radar_called': 'radar_tool' in called_tools,
            'fusion_called': 'fusion_tool' in called_tools,
            'front_debug_called': 'front_vehicle_debug_tool' in called_tools,
            'lane_check_called': 'lane_check_tool' in called_tools,
            'camera_possible_front_vehicle': camera.get('possible_front_vehicle', False),
            'camera_confidence': camera.get('confidence', 0.0),
            'camera_roi_edge_density': camera.get('roi_edge_density', 0.0),
            'camera_roi_dark_ratio': camera.get('roi_dark_ratio', 0.0),
            'lidar_front_distance': lidar.get('front_obstacle_distance', 999.0),
            'lidar_point_count': lidar.get('point_count', 0),
            'lidar_candidate_count_pos': lidar.get('candidate_count_pos', 0),
            'lidar_candidate_count_neg': lidar.get('candidate_count_neg', 0),
            'lidar_selected_bin_point_count': lidar.get('selected_bin_point_count', 0),
            'lidar_front_axis': lidar.get('front_axis', ''),
            'radar_front_distance': radar.get('front_object_distance', 999.0),
            'radar_relative_velocity': radar.get('front_object_relative_velocity', 0.0),
            'radar_ttc': radar.get('ttc', 99.0),
            'radar_point_count': radar.get('radar_point_count', 0),
            'radar_roi_point_count': radar.get('radar_roi_point_count', 0),
            'radar_raw_point_count': radar.get('radar_raw_point_count', 0),
            'radar_selected_bin_point_count': radar.get('selected_bin_point_count', 0),
            'debug_front_vehicle_detected': front_debug.get('front_vehicle_detected', False),
            'debug_front_vehicle_distance': front_debug.get('front_vehicle_distance', 999.0),
            'debug_front_vehicle_speed': front_debug.get('front_vehicle_speed', 0.0),
            'debug_front_vehicle_relative_speed': front_debug.get('relative_speed', 0.0),
            'left_lane_exists': lane.get('left_lane_exists', False),
            'left_lane_clear': lane.get('left_lane_clear', False),
            'left_front_gap': lane.get('left_front_gap', 999.0),
            'left_rear_gap': lane.get('left_rear_gap', 999.0),
            'right_lane_exists': lane.get('right_lane_exists', False),
            'right_lane_clear': lane.get('right_lane_clear', False),
            'right_front_gap': lane.get('right_front_gap', 999.0),
            'right_rear_gap': lane.get('right_rear_gap', 999.0),
            'fusion_front_distance': fusion.get('front_vehicle_distance', 999.0),
            'fusion_ttc': fusion.get('ttc', 99.0),
            'fusion_confidence': fusion.get('confidence', 0.0),
            'final_front_distance': decision.front_vehicle_distance if decision else 999.0,
            'risk_level': decision.risk_level if decision else '',
            'driving_advice': decision.driving_advice if decision else '',
            'target_speed_advice': decision.target_speed_advice if decision else '',
            'maneuver': decision.maneuver if decision else '',
            'target_lane': decision.target_lane if decision else '',
            'lane_change_required': decision.lane_change_required if decision else False,
            'overtake_state': _safe_getattr(agent, 'overtake_state', ''),
            'original_lane_id': _safe_getattr(agent, 'original_lane_id', 0),
            'current_lane_id': _safe_getattr(agent, '_current_lane_id', lambda: 0)(),
            'target_lane_id': _safe_getattr(agent, 'target_lane_id', 0),
            'tracked_front_actor_id': _safe_getattr(agent, 'tracked_front_actor_id', -1),
            'passed_front_vehicle': _safe_getattr(agent, 'passed_front_vehicle', False),
            'return_lane_clear': _safe_getattr(agent, 'return_lane_clear', False),
            'maneuver_applied': _safe_getattr(agent, 'last_maneuver_applied', ''),
            'maneuver_reason': _safe_getattr(agent, 'last_maneuver_reason', ''),
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
