# -*- coding: utf-8 -*-
"""Safety shield for LLM Agent driving advice."""

from opencda.customize.core.llm_agent.response_parser import LLMDecision


class SafetyShield(object):
    """
    Override unsafe LLM decisions using hard safety thresholds.

    The shield uses only ego self-perceived tool outputs. To avoid persistent
    false emergency braking from single radar clutter points, radar-only
    distance does not trigger a critical override unless its TTC is also risky.
    """

    def __init__(self, medium_distance=60.0, high_distance=30.0,
                 critical_distance=12.0):
        self.medium_distance = float(medium_distance)
        self.high_distance = float(high_distance)
        self.critical_distance = float(critical_distance)
        self.critical_ttc = 2.5
        self.high_ttc = 4.5
        self.min_radar_confidence = 0.35

    def _override(self, risk, distance, advice, speed, reason):
        return LLMDecision(
            tools_to_call_next=['fusion_tool'],
            fusion_required=True,
            risk_level=risk,
            front_vehicle_distance=distance,
            driving_advice=advice,
            target_speed_advice=speed,
            reason=reason
        )

    def apply(self, decision, tool_results):
        lidar = tool_results.get('lidar_tool', {}) or {}
        radar = tool_results.get('radar_tool', {}) or {}

        lidar_detected = bool(lidar.get('front_obstacle_detected', False))
        lidar_distance = float(lidar.get('front_obstacle_distance', 999.0))
        radar_detected = bool(radar.get('front_object_detected', False))
        radar_distance = float(radar.get('front_object_distance', 999.0))
        radar_ttc = float(radar.get('ttc', 99.0))
        radar_conf = float(radar.get('confidence', 0.0))

        if lidar_detected and lidar_distance < self.critical_distance:
            return self._override(
                'critical', lidar_distance, 'emergency_slow',
                min(decision.target_speed_advice, 5.0),
                'SafetyShield override: LiDAR critical front distance.')

        if (radar_detected and radar_conf >= self.min_radar_confidence and
                radar_ttc < self.critical_ttc):
            return self._override(
                'critical', radar_distance, 'emergency_slow',
                min(decision.target_speed_advice, 5.0),
                'SafetyShield override: radar TTC is critical.')

        if lidar_detected and lidar_distance < self.high_distance and \
                decision.risk_level in ['low', 'medium']:
            return self._override(
                'high', lidar_distance, 'slow_down',
                min(decision.target_speed_advice, 15.0),
                'SafetyShield override: LiDAR high-risk front distance.')

        if (radar_detected and radar_conf >= self.min_radar_confidence and
                radar_ttc < self.high_ttc and
                decision.risk_level in ['low', 'medium']):
            return self._override(
                'high', radar_distance, 'slow_down',
                min(decision.target_speed_advice, 15.0),
                'SafetyShield override: radar TTC is high-risk.')

        if lidar_detected and lidar_distance < self.medium_distance and \
                decision.risk_level == 'low':
            return self._override(
                'medium', lidar_distance, 'slow_down',
                min(decision.target_speed_advice, 30.0),
                'SafetyShield override: LiDAR medium-risk front distance.')

        return decision
