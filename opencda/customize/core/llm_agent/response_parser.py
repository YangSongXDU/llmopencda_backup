# -*- coding: utf-8 -*-
"""Parse and validate LLM Agent JSON responses."""

import json


class LLMDecision(object):
    """Structured high-level decision produced by the LLM Sensor Agent."""

    VALID_RISK = ['low', 'medium', 'high', 'critical']
    VALID_LEVEL = ['low', 'medium', 'high']
    VALID_MANEUVER = [
        'keep_lane',
        'follow_front_vehicle',
        'overtake_left',
        'overtake_right',
        'abort_overtake'
    ]
    VALID_TARGET_LANE = ['current', 'left', 'right']

    def __init__(self,
                 tools_to_call_next=None,
                 fusion_required=False,
                 risk_level='low',
                 front_vehicle_distance=999.0,
                 driving_advice='keep_speed',
                 target_speed_advice=50.0,
                 maneuver='keep_lane',
                 target_lane='current',
                 lane_change_required=False,
                 reason='',
                 tool_selection_reason='',
                 uncertainty_level='medium',
                 expected_information_gain='medium',
                 fusion_trigger_reason='',
                 resource_budget_level='medium',
                 safety_evidence='insufficient'):
        self.tools_to_call_next = list(tools_to_call_next or [])
        self.fusion_required = bool(fusion_required)
        self.risk_level = risk_level if risk_level in self.VALID_RISK else 'low'
        self.front_vehicle_distance = float(front_vehicle_distance)
        self.driving_advice = driving_advice
        self.target_speed_advice = float(target_speed_advice)
        self.maneuver = maneuver if maneuver in self.VALID_MANEUVER else 'keep_lane'
        self.target_lane = target_lane if target_lane in self.VALID_TARGET_LANE else 'current'
        self.lane_change_required = bool(lane_change_required)
        self.reason = reason
        self.tool_selection_reason = tool_selection_reason
        self.uncertainty_level = (
            uncertainty_level if uncertainty_level in self.VALID_LEVEL else 'medium')
        self.expected_information_gain = (
            expected_information_gain
            if expected_information_gain in self.VALID_LEVEL else 'medium')
        self.fusion_trigger_reason = fusion_trigger_reason
        self.resource_budget_level = (
            resource_budget_level
            if resource_budget_level in self.VALID_LEVEL else 'medium')
        self.safety_evidence = safety_evidence

    def to_dict(self):
        return {
            'tools_to_call_next': self.tools_to_call_next,
            'fusion_required': self.fusion_required,
            'risk_level': self.risk_level,
            'front_vehicle_distance': self.front_vehicle_distance,
            'driving_advice': self.driving_advice,
            'target_speed_advice': self.target_speed_advice,
            'maneuver': self.maneuver,
            'target_lane': self.target_lane,
            'lane_change_required': self.lane_change_required,
            'tool_selection_reason': self.tool_selection_reason,
            'uncertainty_level': self.uncertainty_level,
            'expected_information_gain': self.expected_information_gain,
            'fusion_trigger_reason': self.fusion_trigger_reason,
            'resource_budget_level': self.resource_budget_level,
            'safety_evidence': self.safety_evidence,
            'reason': self.reason
        }


class LLMResponseParser(object):
    """Parse JSON text into LLMDecision."""

    @staticmethod
    def _extract_json(text):
        if not isinstance(text, str):
            return text
        s = text.strip()
        if s.startswith('```'):
            lines = s.splitlines()
            if len(lines) >= 3:
                s = '\n'.join(lines[1:-1]).strip()
                if s.lower().startswith('json'):
                    s = s[4:].strip()
        if not s.startswith('{'):
            start = s.find('{')
            end = s.rfind('}')
            if start >= 0 and end > start:
                s = s[start:end + 1]
        return s

    @staticmethod
    def _filter_tools(tools, allowed_tools=None):
        allowed = set(allowed_tools or [])
        filtered = []
        if not isinstance(tools, list):
            return filtered
        for tool_name in tools:
            if not isinstance(tool_name, str):
                continue
            if allowed and tool_name not in allowed:
                continue
            if tool_name not in filtered:
                filtered.append(tool_name)
        return filtered

    @staticmethod
    def parse(text, fallback_decision=None, allowed_tools=None):
        if isinstance(text, dict):
            data = text
        else:
            try:
                data = json.loads(LLMResponseParser._extract_json(text))
            except Exception:
                return fallback_decision or LLMDecision(
                    tools_to_call_next=[],
                    fusion_required=False,
                    risk_level='low',
                    uncertainty_level='medium',
                    resource_budget_level='low',
                    reason='LLM response parse failed; fallback to low-cost policy.')

        return LLMDecision(
            tools_to_call_next=LLMResponseParser._filter_tools(
                data.get('tools_to_call_next', []), allowed_tools),
            fusion_required=data.get('fusion_required', False),
            risk_level=data.get('risk_level', 'low'),
            front_vehicle_distance=data.get('front_vehicle_distance', 999.0),
            driving_advice=data.get('driving_advice', 'keep_speed'),
            target_speed_advice=data.get('target_speed_advice', 50.0),
            maneuver=data.get('maneuver', 'keep_lane'),
            target_lane=data.get('target_lane', 'current'),
            lane_change_required=data.get('lane_change_required', False),
            tool_selection_reason=data.get('tool_selection_reason', ''),
            uncertainty_level=data.get('uncertainty_level', 'medium'),
            expected_information_gain=data.get(
                'expected_information_gain', 'medium'),
            fusion_trigger_reason=data.get('fusion_trigger_reason', ''),
            resource_budget_level=data.get('resource_budget_level', 'medium'),
            safety_evidence=data.get('safety_evidence', 'insufficient'),
            reason=data.get('reason', '')
        )
