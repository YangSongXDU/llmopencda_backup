# -*- coding: utf-8 -*-
"""Parse and validate LLM Agent JSON responses."""

import json


class LLMDecision(object):
    """Structured high-level decision produced by the LLM Sensor Agent."""

    VALID_RISK = ['low', 'medium', 'high', 'critical']
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
                 reason=''):
        self.tools_to_call_next = tools_to_call_next or []
        self.fusion_required = bool(fusion_required)
        self.risk_level = risk_level if risk_level in self.VALID_RISK else 'low'
        self.front_vehicle_distance = float(front_vehicle_distance)
        self.driving_advice = driving_advice
        self.target_speed_advice = float(target_speed_advice)
        self.maneuver = maneuver if maneuver in self.VALID_MANEUVER else 'keep_lane'
        self.target_lane = target_lane if target_lane in self.VALID_TARGET_LANE else 'current'
        self.lane_change_required = bool(lane_change_required)
        self.reason = reason

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
            'reason': self.reason
        }


class LLMResponseParser(object):
    """Parse JSON text into LLMDecision."""

    @staticmethod
    def parse(text, fallback_decision=None):
        if isinstance(text, dict):
            data = text
        else:
            try:
                data = json.loads(text)
            except Exception:
                return fallback_decision or LLMDecision(
                    risk_level='low',
                    reason='LLM response parse failed; fallback to low risk.')

        return LLMDecision(
            tools_to_call_next=data.get('tools_to_call_next', []),
            fusion_required=data.get('fusion_required', False),
            risk_level=data.get('risk_level', 'low'),
            front_vehicle_distance=data.get('front_vehicle_distance', 999.0),
            driving_advice=data.get('driving_advice', 'keep_speed'),
            target_speed_advice=data.get('target_speed_advice', 50.0),
            maneuver=data.get('maneuver', 'keep_lane'),
            target_lane=data.get('target_lane', 'current'),
            lane_change_required=data.get('lane_change_required', False),
            reason=data.get('reason', '')
        )
