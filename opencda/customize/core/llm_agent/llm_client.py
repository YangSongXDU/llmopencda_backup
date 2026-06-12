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
        distance = float(lidar.get('front_obstacle_distance', 999.0))
        detected = bool(lidar.get('front_obstacle_detected', False))

        if detected and distance < self.critical_distance:
            decision = {
                'tools_to_call_next': ['fusion_tool'],
                'fusion_required': True,
                'risk_level': 'critical',
                'front_vehicle_distance': distance,
                'driving_advice': 'emergency_slow',
                'target_speed_advice': 5.0,
                'reason': 'LiDAR reports a very close front obstacle.'
            }
        elif detected and distance < self.high_distance:
            decision = {
                'tools_to_call_next': ['fusion_tool'],
                'fusion_required': True,
                'risk_level': 'high',
                'front_vehicle_distance': distance,
                'driving_advice': 'slow_down',
                'target_speed_advice': 15.0,
                'reason': 'LiDAR front distance is within high-risk range.'
            }
        elif detected and distance < self.medium_distance:
            decision = {
                'tools_to_call_next': ['fusion_tool'],
                'fusion_required': True,
                'risk_level': 'medium',
                'front_vehicle_distance': distance,
                'driving_advice': 'slow_down',
                'target_speed_advice': 30.0,
                'reason': 'A front obstacle is detected within medium range.'
            }
        else:
            decision = {
                'tools_to_call_next': [],
                'fusion_required': False,
                'risk_level': 'low',
                'front_vehicle_distance': 999.0,
                'driving_advice': 'keep_speed',
                'target_speed_advice': 50.0,
                'reason': 'No close front obstacle is detected by the called tools.'
            }

        return json.dumps(decision)
