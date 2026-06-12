# -*- coding: utf-8 -*-
"""Safety shield for LLM Agent driving advice."""

from opencda.customize.core.llm_agent.response_parser import LLMDecision


class SafetyShield(object):
    """
    Override unsafe LLM decisions using hard safety thresholds.

    This module runs after the LLM decision and before BehaviorAgent speed
    control. It prevents a slow or unsafe LLM response from causing obvious
    front-collision risk.
    """

    def __init__(self, medium_distance=60.0, high_distance=30.0,
                 critical_distance=12.0):
        self.medium_distance = float(medium_distance)
        self.high_distance = float(high_distance)
        self.critical_distance = float(critical_distance)

    def apply(self, decision, tool_results):
        lidar = tool_results.get('lidar_tool', {}) or {}
        detected = bool(lidar.get('front_obstacle_detected', False))
        distance = float(lidar.get('front_obstacle_distance', 999.0))

        if not detected:
            return decision

        if distance < self.critical_distance:
            return LLMDecision(
                tools_to_call_next=['fusion_tool'],
                fusion_required=True,
                risk_level='critical',
                front_vehicle_distance=distance,
                driving_advice='emergency_slow',
                target_speed_advice=min(decision.target_speed_advice, 5.0),
                reason='SafetyShield override: critical front distance.'
            )

        if distance < self.high_distance and decision.risk_level in ['low', 'medium']:
            return LLMDecision(
                tools_to_call_next=['fusion_tool'],
                fusion_required=True,
                risk_level='high',
                front_vehicle_distance=distance,
                driving_advice='slow_down',
                target_speed_advice=min(decision.target_speed_advice, 15.0),
                reason='SafetyShield override: high-risk front distance.'
            )

        if distance < self.medium_distance and decision.risk_level == 'low':
            return LLMDecision(
                tools_to_call_next=['fusion_tool'],
                fusion_required=True,
                risk_level='medium',
                front_vehicle_distance=distance,
                driving_advice='slow_down',
                target_speed_advice=min(decision.target_speed_advice, 30.0),
                reason='SafetyShield override: medium-risk front distance.'
            )

        return decision
