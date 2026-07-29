# -*- coding: utf-8 -*-
"""Prompt builder for the LLM Sensor Agent."""

import json


class PromptBuilder(object):
    """Build a compact prompt that asks the LLM to output strict JSON."""

    @staticmethod
    def build(ego_state, tool_results, available_tools, constraints,
              tool_metadata=None, previous_tool_results=None,
              tool_budget=None, resource_policy=None):
        payload = {
            'task': (
                'Act as a resource-aware autonomous-driving tool-selection '
                'agent. Select the minimum sufficient sensor tools, decide '
                'whether result-level fusion is needed, estimate risk, and '
                'output a high-level driving maneuver. Do not output throttle, '
                'brake, or steer.'
            ),
            'ego_state': ego_state,
            'available_tools': available_tools,
            'tool_metadata': tool_metadata or {},
            'tool_results': tool_results,
            'previous_tool_results': previous_tool_results or {},
            'tool_budget': tool_budget or {},
            'resource_policy': resource_policy or {
                'select_minimum_sufficient_tools': True,
                'avoid_full_multimodal_fusion_under_low_risk': True,
                'use_expensive_tools_only_when_uncertainty_or_risk_justifies_them': True,
                'prefer_lightweight_tools_for_low_risk_monitoring': True,
                'call_fusion_only_for_uncertainty_or_cross_modal_conflict': True
            },
            'constraints': constraints,
            'tool_selection_policy': {
                'low_risk': 'Use ego state and low-cost tools; do not request full fusion.',
                'medium_uncertainty': 'Request one informative tool such as radar or LiDAR.',
                'high_risk_or_conflict': 'Request stronger ranging tools and fusion if multiple modalities are available.',
                'fusion': 'Use fusion only after at least two upstream modal summaries are available.'
            },
            'maneuver_policy': {
                'keep_lane': 'No reliable front risk or no need to change lane.',
                'follow_front_vehicle': 'A front vehicle blocks ego but no adjacent lane is safe.',
                'overtake_left': 'A slower front vehicle is present and the left lane is clear.',
                'overtake_right': 'A slower front vehicle is present and the right lane is clear.',
                'abort_overtake': 'The requested lane change is unsafe or uncertainty is too high.'
            },
            'required_output_json_schema': {
                'tools_to_call_next': ['tool_name'],
                'fusion_required': 'bool',
                'tool_selection_reason': 'short reason for selected tools',
                'uncertainty_level': 'low|medium|high',
                'expected_information_gain': 'low|medium|high',
                'fusion_trigger_reason': 'short reason, empty if fusion is not needed',
                'resource_budget_level': 'low|medium|high',
                'risk_level': 'low|medium|high|critical',
                'front_vehicle_distance': 'float meters, 999 if unknown',
                'driving_advice': 'keep_speed|slow_down|emergency_slow',
                'target_speed_advice': 'float km/h',
                'maneuver': 'keep_lane|follow_front_vehicle|overtake_left|overtake_right|abort_overtake',
                'target_lane': 'current|left|right',
                'lane_change_required': 'bool',
                'reason': 'short explanation'
            }
        }
        return json.dumps(payload, indent=2)
