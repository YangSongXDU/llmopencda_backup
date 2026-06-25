# -*- coding: utf-8 -*-
"""
LLM client abstraction for the OpenCDA LLM Agent demo.

The default offline mode is intentionally deterministic so the demo can run
without an API key. Later, replace LocalHeuristicLLMClient with an OpenAI,
DeepSeek, Ollama, or other provider client.
"""

import json


class LocalHeuristicLLMClient(object):
    """Offline deterministic stand-in for an LLM."""

    def __init__(self, medium_distance=60.0, high_distance=30.0,
                 critical_distance=12.0):
        self.medium_distance = float(medium_distance)
        self.high_distance = float(high_distance)
        self.critical_distance = float(critical_distance)
        self.high_ttc = 4.5
        self.critical_ttc = 2.5

    def _decision(self, risk, distance, advice, speed, fusion, reason):
        return {
            'tools_to_call_next': ['fusion_tool'] if fusion else [],
            'fusion_required': bool(fusion),
            'risk_level': risk,
            'front_vehicle_distance': distance,
            'driving_advice': advice,
            'target_speed_advice': speed,
            'reason': reason
        }

    def complete(self, prompt):
        try:
            payload = json.loads(prompt)
        except Exception:
            payload = {}

        tool_results = payload.get('tool_results', {}) or {}
        lidar = tool_results.get('lidar_tool', {}) or {}
        radar = tool_results.get('radar_tool', {}) or {}
        camera = tool_results.get('camera_tool', {}) or {}

        lidar_detected = bool(lidar.get('front_obstacle_detected', False))
        lidar_distance = float(lidar.get('front_obstacle_distance', 999.0))
        radar_detected = bool(radar.get('front_object_detected', False))
        radar_distance = float(radar.get('front_object_distance', 999.0))
        radar_ttc = float(radar.get('ttc', 99.0))
        radar_conf = float(radar.get('confidence', 0.0))
        camera_possible = bool(camera.get('possible_front_vehicle', False))
        camera_conf = float(camera.get('confidence', 0.0))

        # LiDAR provides geometric distance. Radar-only distance is not enough
        # for a critical decision because single radar returns can be clutter.
        if lidar_detected:
            distance = lidar_distance
        elif radar_detected and radar_ttc < self.high_ttc:
            distance = radar_distance
        else:
            distance = 999.0

        if lidar_detected and lidar_distance < self.critical_distance:
            return json.dumps(self._decision(
                'critical', lidar_distance, 'emergency_slow', 5.0, True,
                'LiDAR self-perception reports a critical front distance.'))

        if radar_detected and radar_ttc < self.critical_ttc and radar_conf >= 0.35:
            return json.dumps(self._decision(
                'critical', radar_distance, 'emergency_slow', 5.0, True,
                'Radar self-perception reports a critical TTC.'))

        if lidar_detected and lidar_distance < self.high_distance:
            return json.dumps(self._decision(
                'high', lidar_distance, 'slow_down', 15.0, True,
                'LiDAR self-perception reports a high-risk front distance.'))

        if radar_detected and radar_ttc < self.high_ttc and radar_conf >= 0.35:
            return json.dumps(self._decision(
                'high', radar_distance, 'slow_down', 15.0, True,
                'Radar self-perception reports a high-risk TTC.'))

        if lidar_detected and lidar_distance < self.medium_distance:
            return json.dumps(self._decision(
                'medium', lidar_distance, 'slow_down', 30.0, True,
                'LiDAR self-perception reports a medium-range front object.'))

        # Low-cost confirmation: call fusion only when there is at least a weak
        # visual cue and one range sensor reports something uncertain.
        weak_multimodal_cue = (
            camera_possible and camera_conf >= 0.45 and
            (lidar_detected or (radar_detected and radar_conf >= 0.50))
        )
        if weak_multimodal_cue:
            return json.dumps(self._decision(
                'low', distance, 'keep_speed', 50.0, True,
                'Weak multimodal cue exists; call fusion for confirmation.'))

        return json.dumps(self._decision(
            'low', 999.0, 'keep_speed', 50.0, False,
            'No reliable close front object is detected by ego sensor tools.'))
