# -*- coding: utf-8 -*-
"""CSV recorder for LLM Sensor Tool Agent demos."""

import csv
import os

from opencda.customize.core.common.resource_monitor import \
    ProcessResourceMonitor


def _safe_getattr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


class LLMExperimentRecorder(object):
    """Record tool calls, LLM decisions, maneuvers, and controls."""

    def __init__(self, save_path, resource_monitor_config=None):
        self.save_path = save_path
        self.rows = []
        self.resource_monitor = ProcessResourceMonitor(resource_monitor_config)
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.fieldnames = [
            'step',
            'ego_x', 'ego_y', 'ego_z', 'ego_yaw', 'ego_speed',
            'called_tools', 'tool_call_count', 'tool_total_cost',
            'resource_counted_tools', 'resource_counted_tool_count',
            'requested_tools', 'executed_tools', 'skipped_tools',
            'cached_tools',
            'tool_selection_reason', 'uncertainty_level',
            'expected_information_gain', 'fusion_trigger_reason',
            'resource_budget_level',
            'tool_budget', 'tool_budget_used', 'tool_budget_exceeded',
            'total_tool_runtime_ms',
            'oracle_tool_runtime_ms', 'oracle_tool_cost',
            'agent_cycle_runtime_ms',
            'camera_runtime_ms', 'lidar_runtime_ms', 'radar_runtime_ms',
            'fusion_runtime_ms', 'safety_evidence', 'oracle_tool_used',
            'llm_backend', 'llm_fallback_used', 'llm_error',
            'llm_call_executed', 'llm_request_latency_ms',
            'llm_retry_count', 'llm_http_status', 'llm_response_valid',
            'llm_call_success', 'llm_fallback_reason', 'llm_circuit_open',
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
            'left_lane_continuous_distance',
            'left_lane_ends_or_merges_ahead', 'left_overtake_suitable',
            'right_lane_exists', 'right_lane_clear', 'right_front_gap', 'right_rear_gap',
            'right_lane_continuous_distance',
            'right_lane_ends_or_merges_ahead', 'right_overtake_suitable',
            'current_lane_continuous_distance',
            'current_lane_ends_or_merges_ahead',
            'fusion_front_distance', 'fusion_ttc', 'fusion_confidence',
            'final_front_distance',
            'risk_level', 'driving_advice', 'target_speed_advice',
            'maneuver', 'target_lane', 'lane_change_required',
            'overtake_state', 'original_lane_id', 'current_lane_id',
            'target_lane_id', 'tracked_front_actor_id',
            'overtake_attempt_count', 'completed_overtake_count',
            'overtake_abort_count', 'last_overtake_abort_reason',
            'lane_change_steps', 'passing_steps',
            'overtake_cooldown',
            'passed_front_vehicle', 'return_lane_clear',
            'maneuver_applied', 'maneuver_reason',
            'target_speed', 'throttle', 'brake', 'steer',
            'client_process_cpu_percent', 'client_process_rss_mb',
            'host_cpu_percent', 'host_memory_used_mb',
            'gpu_utilization_percent', 'gpu_memory_used_mb',
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
        oracle_results = _safe_getattr(
            llm_agent, 'last_oracle_results', {}) or {}
        called_tools = _safe_getattr(llm_agent, 'last_called_tools', []) or []
        resource_counted_tools = _safe_getattr(
            llm_agent, 'last_resource_counted_tools', []) or []
        requested_tools = _safe_getattr(llm_agent, 'last_requested_tools', []) or []
        executed_tools = _safe_getattr(llm_agent, 'last_executed_tools', []) or []
        skipped_tools = _safe_getattr(llm_agent, 'last_skipped_tools', []) or []
        cached_tools = _safe_getattr(llm_agent, 'last_cached_tools', []) or []
        runtime_ms = _safe_getattr(llm_agent, 'last_tool_runtime_ms', {}) or {}

        camera = tool_results.get('camera_tool', {}) or {}
        lidar = tool_results.get('lidar_tool', {}) or {}
        radar = tool_results.get('radar_tool', {}) or {}
        front_debug = tool_results.get('front_vehicle_debug_tool', {}) or \
            oracle_results.get('front_vehicle_debug_tool', {}) or {}
        lane = tool_results.get('lane_check_tool', {}) or \
            oracle_results.get('lane_check_tool', {}) or {}
        fusion = tool_results.get('fusion_tool', {}) or {}
        resource_sample = self.resource_monitor.sample()

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
            'resource_counted_tools': '|'.join(resource_counted_tools),
            'resource_counted_tool_count': len(resource_counted_tools),
            'requested_tools': '|'.join(requested_tools),
            'executed_tools': '|'.join(executed_tools),
            'skipped_tools': '|'.join(skipped_tools),
            'cached_tools': '|'.join(cached_tools),
            'tool_selection_reason': _safe_getattr(
                llm_agent, 'last_tool_selection_reason', ''),
            'uncertainty_level': _safe_getattr(
                llm_agent, 'last_uncertainty_level', ''),
            'expected_information_gain': _safe_getattr(
                llm_agent, 'last_expected_information_gain', ''),
            'fusion_trigger_reason': _safe_getattr(
                llm_agent, 'last_fusion_trigger_reason', ''),
            'resource_budget_level': _safe_getattr(
                llm_agent, 'last_resource_budget_level', ''),
            'tool_budget': _safe_getattr(llm_agent, 'last_tool_budget', 0.0),
            'tool_budget_used': _safe_getattr(
                llm_agent, 'last_tool_budget_used', 0.0),
            'tool_budget_exceeded': _safe_getattr(
                llm_agent, 'last_tool_budget_exceeded', False),
            'total_tool_runtime_ms': _safe_getattr(
                llm_agent, 'last_total_tool_runtime_ms', 0.0),
            'oracle_tool_runtime_ms': _safe_getattr(
                llm_agent, 'last_oracle_tool_runtime_ms', 0.0),
            'oracle_tool_cost': _safe_getattr(
                llm_agent, 'last_oracle_tool_cost', 0.0),
            'agent_cycle_runtime_ms': _safe_getattr(
                llm_agent, 'last_agent_cycle_runtime_ms', 0.0),
            'camera_runtime_ms': runtime_ms.get('camera_tool', 0.0),
            'lidar_runtime_ms': runtime_ms.get('lidar_tool', 0.0),
            'radar_runtime_ms': runtime_ms.get('radar_tool', 0.0),
            'fusion_runtime_ms': runtime_ms.get('fusion_tool', 0.0),
            'safety_evidence': _safe_getattr(
                llm_agent, 'last_safety_evidence', ''),
            'oracle_tool_used': _safe_getattr(
                llm_agent, 'last_oracle_tool_used', False),
            'llm_backend': _safe_getattr(llm_agent, 'last_llm_backend', ''),
            'llm_fallback_used': _safe_getattr(llm_agent, 'last_llm_fallback_used', False),
            'llm_error': _safe_getattr(llm_agent, 'last_llm_error', ''),
            'llm_call_executed': _safe_getattr(
                llm_agent, 'last_llm_call_executed', False),
            'llm_request_latency_ms': _safe_getattr(
                llm_agent, 'last_llm_request_latency_ms', 0.0),
            'llm_retry_count': _safe_getattr(
                llm_agent, 'last_llm_retry_count', 0),
            'llm_http_status': _safe_getattr(
                llm_agent, 'last_llm_http_status', 0),
            'llm_response_valid': _safe_getattr(
                llm_agent, 'last_llm_response_valid', False),
            'llm_call_success': _safe_getattr(
                llm_agent, 'last_llm_call_success', False),
            'llm_fallback_reason': _safe_getattr(
                llm_agent, 'last_llm_fallback_reason', ''),
            'llm_circuit_open': _safe_getattr(
                llm_agent, 'last_llm_circuit_open', False),
            'camera_called': 'camera_tool' in executed_tools,
            'lidar_called': 'lidar_tool' in executed_tools,
            'radar_called': 'radar_tool' in executed_tools,
            'fusion_called': 'fusion_tool' in executed_tools,
            'front_debug_called': 'front_vehicle_debug_tool' in executed_tools,
            'lane_check_called': 'lane_check_tool' in executed_tools,
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
            'left_lane_continuous_distance': lane.get(
                'left_lane_continuous_distance', 0.0),
            'left_lane_ends_or_merges_ahead': lane.get(
                'left_lane_ends_or_merges_ahead', False),
            'left_overtake_suitable': lane.get(
                'left_overtake_suitable', False),
            'right_lane_exists': lane.get('right_lane_exists', False),
            'right_lane_clear': lane.get('right_lane_clear', False),
            'right_front_gap': lane.get('right_front_gap', 999.0),
            'right_rear_gap': lane.get('right_rear_gap', 999.0),
            'right_lane_continuous_distance': lane.get(
                'right_lane_continuous_distance', 0.0),
            'right_lane_ends_or_merges_ahead': lane.get(
                'right_lane_ends_or_merges_ahead', False),
            'right_overtake_suitable': lane.get(
                'right_overtake_suitable', False),
            'current_lane_continuous_distance': lane.get(
                'current_lane_continuous_distance', 0.0),
            'current_lane_ends_or_merges_ahead': lane.get(
                'current_lane_ends_or_merges_ahead', False),
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
            'overtake_attempt_count': _safe_getattr(
                agent, 'overtake_attempt_count', 0),
            'completed_overtake_count': _safe_getattr(
                agent, 'completed_overtake_count', 0),
            'overtake_abort_count': _safe_getattr(
                agent, 'overtake_abort_count', 0),
            'last_overtake_abort_reason': _safe_getattr(
                agent, 'last_overtake_abort_reason', ''),
            'lane_change_steps': _safe_getattr(agent, 'lane_change_steps', 0),
            'passing_steps': _safe_getattr(agent, 'passing_steps', 0),
            'overtake_cooldown': _safe_getattr(
                agent, 'overtake_cooldown', 0),
            'passed_front_vehicle': _safe_getattr(agent, 'passed_front_vehicle', False),
            'return_lane_clear': _safe_getattr(agent, 'return_lane_clear', False),
            'maneuver_applied': _safe_getattr(agent, 'last_maneuver_applied', ''),
            'maneuver_reason': _safe_getattr(agent, 'last_maneuver_reason', ''),
            'target_speed': _safe_getattr(agent, 'last_target_speed', ''),
            'throttle': control.throttle,
            'brake': control.brake,
            'steer': control.steer,
            'client_process_cpu_percent': resource_sample[
                'client_process_cpu_percent'],
            'client_process_rss_mb': resource_sample[
                'client_process_rss_mb'],
            'host_cpu_percent': resource_sample['host_cpu_percent'],
            'host_memory_used_mb': resource_sample['host_memory_used_mb'],
            'gpu_utilization_percent': resource_sample[
                'gpu_utilization_percent'],
            'gpu_memory_used_mb': resource_sample['gpu_memory_used_mb'],
            'reason': decision.reason if decision else ''
        }
        self.rows.append(row)

    def save(self):
        try:
            with open(self.save_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
        finally:
            self.resource_monitor.close()
        print('[LLMExperimentRecorder] CSV saved to: %s' % self.save_path)
