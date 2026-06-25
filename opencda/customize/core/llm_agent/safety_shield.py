# -*- coding: utf-8 -*-
"""Safety shield for LLM Agent driving advice."""

from opencda.customize.core.llm_agent.response_parser import LLMDecision


class SafetyShield(object):
    """
    Override unsafe LLM decisions using hard safety thresholds.

    This module runs after the LLM decision and before BehaviorAgent speed
    control. It prevents a slow or unsafe LLM response from causing obvious
    front-collision risk. It uses only ego self-perceived tool outputs.
    """

    def __init__(self, medium_distance=60.0, high_distance=30.0,
                 critical_distance=12.0):
        self.medium_distance = float(medium_distance)
        self.high_distance = float(high_distance)
        self.critical_distance = float(critical_distance)
        self.critical_ttc = 2.5
        self.high_ttc = 4.5

    def apply(self, decision, tool_results):
        lidar = tool_results.get('lidar_tool', {}) or {}
        radar = tool_results.get('radar_tool', {}) or {}

        lidar_detected = bool(lidar.get('front_obstacle_detected', False))
        lidar_distance = float(lidar.get('front_obstacle_distance', 999.0))
        radar_detected = bool(radar.get('front_object_detected', False))
        radar_distance = float(radar.get('front_object_distance', 999.0))
        radar_ttc = float(radar.get('ttc', 99.0))

        detected_distances = []
        if lidar_detected:
            detected_distances.append(lidar_distance)
        if radar_detected:
            detected_distances.append(radar_distance)
        if not detected_distances:
            return decision
        distance = min(detected_distances)

        if radar_detected and radar_ttc < self.critical_ttc:
            return LLMDecision(
                tools_to_call_next=['fusion_tool'],
                fusion_required=True,
                risk_level='critical',
                front_vehicle_distance=distance,
                driving_advice='emergency_slow',
                target_speed_advice=min(decision.target_speed_advice, 5.0),
                reason='SafetyShield override: radar TTC is critical.'
            )

        if distance < self.critical_distance:
            return LLMDecision(
                tools_to_call_next=['fusion_tool'],
                fusion_required=True,
                risk_level='critical',
                front_vehicle_distance=distance,
                driving_advice='emergency_slow',
                target_speed_advice=min(decision.target_speed_advice, 5.0),
                reason='SafetyShield override: critical self-perceived front distance.'
            )

        if radar_detected and radar_ttc < self.high_ttc:
            return LLMDecision(
                tools_to_call_next=['fusion_tool'],
                fusion_required=True,
                risk_level='high',
                front_vehicle_distance=distance,
                driving_advice='slow_down',
                target_speed_advice=min(decision.target_speed_advice, 15.0),
                reason='SafetyShield override: radar TTC is high-risk.'
            )

        if distance < self.high_distance and decision.risk_level in ['low', 'medium']:
            return LLMDecision(
                tools_to_call_next=['fusion_tool'],
                fusion_required=True,
                risk_level='high',
                front_vehicle_distance=distance,
                driving_advice='slow_down',
                target_speed_advice=min(decision.target_speed_advice, 15.0),
                reason='SafetyShield override: high-risk self-perceived front distance.'
            )

        if distance < self.medium_distance and decision.risk_level == 'low':
            return LLMDecision(
                tools_to_call_next=['fusion_tool'],
                fusion_required=True,
                risk_level='medium',
                front_vehicle_distance=distance,
                driving_advice='slow_down',
                target_speed_advice=min(decision.target_speed_advice, 30.0),
                reason='SafetyShield override: medium-risk self-perceived front distance.'
            )

        return decision
