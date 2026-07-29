# -*- coding: utf-8 -*-
"""Safety shield for LLM Agent driving advice."""

from opencda.customize.core.llm_agent.response_parser import LLMDecision


class SafetyShield(object):
    """
    Override unsafe LLM decisions using hard safety thresholds.

    LiDAR distance is treated as a strong geometric cue. Radar-only TTC is
    debounced and requires enough confidence/bin support before it can upgrade a
    low/medium LLM decision. This reduces persistent high-risk states from
    single-frame radar clutter while preserving emergency protection.
    """

    def __init__(self, medium_distance=60.0, high_distance=30.0,
                 critical_distance=12.0):
        self.medium_distance = float(medium_distance)
        self.high_distance = float(high_distance)
        self.critical_distance = float(critical_distance)
        self.critical_ttc = 2.5
        self.high_ttc = 4.0
        self.min_radar_confidence = 0.45
        self.min_radar_bin_points = 5
        self.radar_high_required_streak = 2
        self.radar_critical_required_streak = 2
        self.radar_high_streak = 0
        self.radar_critical_streak = 0
        self.last_safety_evidence = 'insufficient'

    def _override(self, risk, distance, advice, speed, reason, decision=None):
        maneuver = 'follow_front_vehicle'
        target_lane = 'current'
        lane_change_required = False
        tools_to_call_next = ['fusion_tool']
        tool_selection_reason = 'SafetyShield requests fusion for risk confirmation.'
        uncertainty_level = 'high'
        expected_information_gain = 'high'
        fusion_trigger_reason = reason
        resource_budget_level = 'high'
        if decision is not None:
            maneuver = decision.maneuver
            target_lane = decision.target_lane
            lane_change_required = decision.lane_change_required
            tools_to_call_next = list(decision.tools_to_call_next or [])
            if 'fusion_tool' not in tools_to_call_next:
                tools_to_call_next.append('fusion_tool')
            tool_selection_reason = decision.tool_selection_reason
            uncertainty_level = decision.uncertainty_level
            expected_information_gain = decision.expected_information_gain
            fusion_trigger_reason = decision.fusion_trigger_reason or reason
            resource_budget_level = decision.resource_budget_level

        return LLMDecision(
            tools_to_call_next=tools_to_call_next,
            fusion_required=True,
            risk_level=risk,
            front_vehicle_distance=distance,
            driving_advice=advice,
            target_speed_advice=speed,
            maneuver=maneuver,
            target_lane=target_lane,
            lane_change_required=lane_change_required,
            tool_selection_reason=tool_selection_reason,
            uncertainty_level=uncertainty_level,
            expected_information_gain=expected_information_gain,
            fusion_trigger_reason=fusion_trigger_reason,
            resource_budget_level=resource_budget_level,
            safety_evidence=self.last_safety_evidence,
            reason=reason
        )

    def _update_radar_streaks(self, radar_detected, radar_distance, radar_ttc,
                              radar_conf, radar_bin_points):
        radar_supported = (
            radar_detected and
            radar_distance < self.high_distance and
            radar_conf >= self.min_radar_confidence and
            radar_bin_points >= self.min_radar_bin_points
        )

        if radar_supported and radar_ttc < self.high_ttc:
            self.radar_high_streak += 1
        else:
            self.radar_high_streak = 0

        if radar_supported and radar_ttc < self.critical_ttc:
            self.radar_critical_streak += 1
        else:
            self.radar_critical_streak = 0

    def apply(self, decision, tool_results):
        camera = tool_results.get('camera_tool', {}) or {}
        lidar = tool_results.get('lidar_tool', {}) or {}
        radar = tool_results.get('radar_tool', {}) or {}
        fusion = tool_results.get('fusion_tool', {}) or {}

        camera_available = bool(camera.get('image_available', False))
        lidar_available = bool(lidar.get('success', False))
        radar_available = bool(radar.get('success', False))
        lidar_detected = bool(lidar.get('front_obstacle_detected', False))
        lidar_distance = float(lidar.get('front_obstacle_distance', 999.0))
        lidar_bin_points = int(lidar.get('selected_bin_point_count', 0))
        radar_detected = bool(radar.get('front_object_detected', False))
        radar_distance = float(radar.get('front_object_distance', 999.0))
        radar_ttc = float(radar.get('ttc', 99.0))
        radar_conf = float(radar.get('confidence', 0.0))
        radar_bin_points = int(radar.get('selected_bin_point_count', 0))

        self._update_radar_streaks(
            radar_detected, radar_distance, radar_ttc, radar_conf,
            radar_bin_points)

        if fusion.get('front_vehicle_detected', False):
            self.last_safety_evidence = 'fused'
        elif lidar_available and radar_available:
            self.last_safety_evidence = 'lidar_radar'
        elif lidar_available:
            self.last_safety_evidence = 'lidar_only'
        elif radar_available:
            self.last_safety_evidence = 'radar_only'
        elif camera_available:
            self.last_safety_evidence = 'camera_only'
        else:
            self.last_safety_evidence = 'insufficient'
        decision.safety_evidence = self.last_safety_evidence

        if lidar_detected and lidar_distance < self.critical_distance:
            return self._override(
                'critical', lidar_distance, 'emergency_slow',
                min(decision.target_speed_advice, 5.0),
                'SafetyShield override: LiDAR critical front distance.',
                decision=decision)

        if self.radar_critical_streak >= self.radar_critical_required_streak:
            return self._override(
                'critical', radar_distance, 'emergency_slow',
                min(decision.target_speed_advice, 5.0),
                'SafetyShield override: radar TTC is critical after debounce.',
                decision=decision)

        if (lidar_detected and lidar_distance < self.high_distance and
                lidar_bin_points >= 5 and
                decision.risk_level in ['low', 'medium']):
            return self._override(
                'high', lidar_distance, 'slow_down',
                min(decision.target_speed_advice, 15.0),
                'SafetyShield override: LiDAR high-risk front distance.',
                decision=decision)

        if (self.radar_high_streak >= self.radar_high_required_streak and
                decision.risk_level in ['low', 'medium']):
            return self._override(
                'high', radar_distance, 'slow_down',
                min(decision.target_speed_advice, 15.0),
                'SafetyShield override: radar TTC is high-risk after debounce.',
                decision=decision)

        if (lidar_detected and lidar_distance < self.medium_distance and
                lidar_bin_points >= 5 and decision.risk_level == 'low'):
            return self._override(
                'medium', lidar_distance, 'slow_down',
                min(decision.target_speed_advice, 30.0),
                'SafetyShield override: LiDAR medium-risk front distance.',
                decision=decision)

        if (self.last_safety_evidence in ['insufficient', 'camera_only'] and
                decision.uncertainty_level == 'high'):
            requested = list(decision.tools_to_call_next or [])
            for tool_name in ['radar_tool', 'lidar_tool']:
                if tool_name not in requested:
                    requested.append(tool_name)
            decision.tools_to_call_next = requested
            decision.tool_selection_reason = (
                decision.tool_selection_reason or
                'SafetyShield requests stronger ranging tools because safety evidence is insufficient.')
            decision.safety_evidence = self.last_safety_evidence

        return decision
