# -*- coding: utf-8 -*-
"""Prompt builder for the LLM Sensor Agent."""

import json


class PromptBuilder(object):
    """
    Build a compact prompt that asks the LLM to output strict JSON.
    """

    @staticmethod
    def build(ego_state, tool_results, available_tools, constraints):
        payload = {
            'task': (
                'Decide which sensor tools should be called next, whether '
                'selective fusion is needed, and what high-level driving '
                'advice should be given. Do not output throttle, brake, or steer.'
            ),
            'ego_state': ego_state,
            'available_tools': available_tools,
            'tool_results': tool_results,
            'constraints': constraints,
            'required_output_json_schema': {
                'tools_to_call_next': ['tool_name'],
                'fusion_required': 'bool',
                'risk_level': 'low|medium|high|critical',
                'front_vehicle_distance': 'float meters, 999 if unknown',
                'driving_advice': 'keep_speed|slow_down|emergency_slow',
                'target_speed_advice': 'float km/h',
                'reason': 'short explanation'
            }
        }
        return json.dumps(payload, indent=2)
