# -*- coding: utf-8 -*-
"""
LLM client abstraction for the OpenCDA LLM Agent demo.

The default offline mode is intentionally deterministic so the demo can run
without an API key. Later, replace LocalHeuristicLLMClient with an OpenAI,
DeepSeek, Ollama, or other provider client.
"""

import json


class LocalHeuristicLLMClient(object):
    """
    Offline deterministic stand-in for an LLM.

    It consumes the same prompt that a real LLM would receive and returns a
    JSON string with the same schema. This makes the project runnable first,
    while keeping the LLM-Agent interface stable.
    """

    def __init__(self, medium_distance=60.0, high_distance=30.0,
                 critical_distance=12.0):
        self.medium_distance = float(medium_distance)
        self.high_distance = float(high_distance)
        self.critical_distance = float(critical_distance)

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
        camera_possible = bool(camera.get('possible_front_vehicle', False))

        detected_distances = []
        if lidar_detected:
            detected_distances.append(lidar_distance)
        if radar_detected:
            detected_distances.append(radar_distance)
        distance = min(detected_distances) if detected_distances else 999.0
        detected = len(detected_distances) > 0

        fusion_needed = False
        fusion_reason = []
        if lidar_detected or radar_detected:
            fusion_needed = True
            fusion_reason.append('front object reported by self-perception tools')
        if camera_possible:
            fusion_needed = True
            fusion_reason.append('camera ROI contains a possible front object')
        if lidar_detected and radar_detected and abs(lidar_distance - radar_distance) > 8.0:
            fusion_needed = True
            fusion_reason.append('LiDAR and radar distances are inconsistent')

        if radar_detected and radar_ttc < 2.5:
            decision = {
                'tools_to_call_next': ['fusion_tool'],
                'fusion_required': True,
                'risk_level': 'critical',
                'front_vehicle_distance': distance,
                'driving_advice': 'emergency_slow',
                'target_speed_advice': 5.0,
                'reason': 'Radar TTC is below 2.5 seconds based on ego radar perception.'
            }
        elif detected and distance < self.critical_distance:
            decision = {
                'tools_to_call_next': ['fusion_tool'],
                'fusion_required': True,
                'risk_level': 'critical',
                'front_vehicle_distance': distance,
                'driving_advice': 'emergency_slow',
                'target_speed_advice': 5.0,
                'reason': 'Self-perceived front object is within critical distance.'
            }
        elif radar_detected and radar_ttc < 4.5:
            decision = {
                'tools_to_call_next': ['fusion_tool'],
                'fusion_required': True,
                'risk_level': 'high',
                'front_vehicle_distance': distance,
                'driving_advice': 'slow_down',
                'target_speed_advice': 15.0,
                'reason': 'Radar TTC indicates high closing-risk from self-perceived radar data.'
            }
        elif detected and distance < self.high_distance:
            decision = {
                'tools_to_call_next': ['fusion_tool'],
                'fusion_required': True,
                'risk_level': 'high',
                'front_vehicle_distance': distance,
                'driving_advice': 'slow_down',
                'target_speed_advice': 15.0,
                'reason': 'Self-perceived front object is within high-risk range.'
            }
        elif detected and distance < self.medium_distance:
            decision = {
                'tools_to_call_next': ['fusion_tool'],
                'fusion_required': True,
                'risk_level': 'medium',
                'front_vehicle_distance': distance,
                'driving_advice': 'slow_down',
                'target_speed_advice': 30.0,
                'reason': 'Self-perceived front object is within medium-risk range.'
            }
        else:
            decision = {
                'tools_to_call_next': ['fusion_tool'] if fusion_needed else [],
                'fusion_required': fusion_needed,
                'risk_level': 'low',
                'front_vehicle_distance': 999.0,
                'driving_advice': 'keep_speed',
                'target_speed_advice': 50.0,
                'reason': 'No close front object is detected by ego sensor tools.' if not fusion_reason else '; '.join(fusion_reason)
            }

        return json.dumps(decision)
