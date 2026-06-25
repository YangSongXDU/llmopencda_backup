# -*- coding: utf-8 -*-
"""
LLM client abstraction for the OpenCDA LLM Agent demo.

The default offline mode is deterministic so the demo can run without an API
key. A DeepSeek/OpenAI-compatible chat client is also provided for real LLM
calls when an API key is configured.
"""

import json
import os
import urllib.request


class LocalHeuristicLLMClient(object):
    """Offline deterministic stand-in for an LLM with overtaking behavior."""

    def __init__(self, medium_distance=60.0, high_distance=30.0,
                 critical_distance=12.0):
        self.medium_distance = float(medium_distance)
        self.high_distance = float(high_distance)
        self.critical_distance = float(critical_distance)
        # Keep radar-only risk conservative. Radar is useful for TTC, but in
        # this CARLA setup it can see clutter or adjacent-lane vehicles. A local
        # fallback should therefore be less aggressive than the safety shield.
        self.high_ttc = 3.5
        self.critical_ttc = 2.0
        self.min_radar_confidence = 0.45
        self.min_radar_bin_points = 5
        self.overtake_distance = 45.0
        self.last_backend = 'local'
        self.last_fallback_used = False
        self.last_error = ''

    def _decision(self, risk, distance, advice, speed, fusion, reason,
                  maneuver='keep_lane', target_lane='current',
                  lane_change_required=False):
        return {
            'tools_to_call_next': ['fusion_tool'] if fusion else [],
            'fusion_required': bool(fusion),
            'risk_level': risk,
            'front_vehicle_distance': distance,
            'driving_advice': advice,
            'target_speed_advice': speed,
            'maneuver': maneuver,
            'target_lane': target_lane,
            'lane_change_required': bool(lane_change_required),
            'reason': reason
        }

    @staticmethod
    def _data(tool_result):
        if not isinstance(tool_result, dict):
            return {}
        # ToolResult.to_dict() stores payload fields at the top level in this
        # project. If a future implementation nests them in data, support both.
        nested = tool_result.get('data', None)
        if isinstance(nested, dict):
            merged = dict(tool_result)
            merged.update(nested)
            return merged
        return tool_result

    def complete(self, prompt):
        self.last_backend = 'local'
        self.last_fallback_used = False
        self.last_error = ''

        try:
            payload = json.loads(prompt)
        except Exception:
            payload = {}

        tool_results = payload.get('tool_results', {}) or {}
        lidar = self._data(tool_results.get('lidar_tool', {}) or {})
        radar = self._data(tool_results.get('radar_tool', {}) or {})
        camera = self._data(tool_results.get('camera_tool', {}) or {})
        front_debug = self._data(tool_results.get('front_vehicle_debug_tool', {}) or {})
        lane = self._data(tool_results.get('lane_check_tool', {}) or {})

        lidar_detected = bool(lidar.get('front_obstacle_detected', False))
        lidar_distance = float(lidar.get('front_obstacle_distance', 999.0))
        radar_detected = bool(radar.get('front_object_detected', False))
        radar_distance = float(radar.get('front_object_distance', 999.0))
        radar_ttc = float(radar.get('ttc', 99.0))
        radar_conf = float(radar.get('confidence', 0.0))
        radar_bin_points = int(radar.get('selected_bin_point_count', 0))
        camera_possible = bool(camera.get('possible_front_vehicle', False))
        camera_conf = float(camera.get('confidence', 0.0))

        debug_front_detected = bool(front_debug.get('front_vehicle_detected', False))
        debug_front_distance = float(front_debug.get('front_vehicle_distance', 999.0))
        left_clear = bool(lane.get('left_lane_exists', False) and lane.get('left_lane_clear', False))
        right_clear = bool(lane.get('right_lane_exists', False) and lane.get('right_lane_clear', False))

        # First overtaking prototype: use the debug front-vehicle and lane-check
        # tools when enabled. They make the maneuver demo deterministic and also
        # expose why ego may see distance=999 from pure sensor tools.
        if debug_front_detected and debug_front_distance < self.overtake_distance:
            if left_clear:
                return json.dumps(self._decision(
                    'medium', debug_front_distance, 'keep_speed', 40.0, True,
                    'A slower front vehicle is present and the left lane is clear; request overtake_left.',
                    maneuver='overtake_left', target_lane='left',
                    lane_change_required=True))
            if right_clear:
                return json.dumps(self._decision(
                    'medium', debug_front_distance, 'keep_speed', 35.0, True,
                    'A slower front vehicle is present and the right lane is clear; request overtake_right.',
                    maneuver='overtake_right', target_lane='right',
                    lane_change_required=True))
            return json.dumps(self._decision(
                'high', debug_front_distance, 'slow_down', 18.0, True,
                'A front vehicle is present but no adjacent lane is clear; follow the front vehicle.',
                maneuver='follow_front_vehicle', target_lane='current',
                lane_change_required=False))

        radar_reliable_ttc = (
            radar_detected and radar_distance < self.high_distance and
            radar_conf >= self.min_radar_confidence and
            radar_bin_points >= self.min_radar_bin_points
        )

        # Sensor-only safety logic.
        if lidar_detected:
            distance = lidar_distance
        elif radar_reliable_ttc and radar_ttc < self.high_ttc:
            distance = radar_distance
        else:
            distance = 999.0

        if lidar_detected and lidar_distance < self.critical_distance:
            return json.dumps(self._decision(
                'critical', lidar_distance, 'emergency_slow', 5.0, True,
                'LiDAR self-perception reports a critical front distance.',
                maneuver='follow_front_vehicle'))

        if radar_reliable_ttc and radar_ttc < self.critical_ttc:
            return json.dumps(self._decision(
                'critical', radar_distance, 'emergency_slow', 5.0, True,
                'Radar self-perception reports a critical TTC.',
                maneuver='follow_front_vehicle'))

        if lidar_detected and lidar_distance < self.high_distance:
            return json.dumps(self._decision(
                'high', lidar_distance, 'slow_down', 15.0, True,
                'LiDAR self-perception reports a high-risk front distance.',
                maneuver='follow_front_vehicle'))

        if radar_reliable_ttc and radar_ttc < self.high_ttc:
            return json.dumps(self._decision(
                'high', radar_distance, 'slow_down', 15.0, True,
                'Radar self-perception reports a high-risk TTC.',
                maneuver='follow_front_vehicle'))

        if lidar_detected and lidar_distance < self.medium_distance:
            return json.dumps(self._decision(
                'medium', lidar_distance, 'slow_down', 30.0, True,
                'LiDAR self-perception reports a medium-range front object.',
                maneuver='follow_front_vehicle'))

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


class ChatLLMClient(object):
    """OpenAI-compatible chat-completions client, e.g. DeepSeek."""

    def __init__(self, config=None, fallback_client=None):
        config = config or {}
        self.provider = str(config.get('llm_provider', 'deepseek'))
        self.model = str(config.get('llm_model', 'deepseek-v4-flash'))
        raw_base_url = str(config.get('llm_base_url', 'https://api.deepseek.com'))
        self.base_url = self._normalize_chat_url(raw_base_url)
        self.api_key_env = str(config.get('api_key_env', 'DEEPSEEK_API_KEY'))
        self.timeout = float(config.get('llm_timeout', 20.0))
        self.temperature = float(config.get('temperature', 0.0))
        self.fallback_client = fallback_client
        self.last_backend = self.provider
        self.last_fallback_used = False
        self.last_error = ''

    @staticmethod
    def _normalize_chat_url(base_url):
        """
        Accept either an SDK-style base URL or a full chat-completions URL.

        DeepSeek documentation calls https://api.deepseek.com the OpenAI
        base_url, while raw urllib requests must POST to
        /chat/completions. Normalizing here avoids HTTP 404 when users copy the
        SDK base_url into YAML.
        """
        url = str(base_url).strip().rstrip('/')
        if url.endswith('/chat/completions'):
            return url
        return url + '/chat/completions'

    def complete(self, prompt):
        self.last_backend = self.provider
        self.last_fallback_used = False
        self.last_error = ''

        api_key = os.environ.get(self.api_key_env, '')
        if not api_key:
            self.last_error = 'Missing API key environment variable: %s' % self.api_key_env
            if self.fallback_client is not None:
                text = self.fallback_client.complete(prompt)
                self.last_backend = 'local_fallback_missing_key'
                self.last_fallback_used = True
                return text
            raise RuntimeError(self.last_error)

        body = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        'You are a high-level autonomous-driving decision agent. '
                        'Return strict JSON only. Do not output throttle, brake, or steer.'
                    )
                },
                {'role': 'user', 'content': prompt}
            ],
            'temperature': self.temperature,
            'stream': False
        }
        data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(
            self.base_url,
            data=data,
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer %s' % api_key
            },
            method='POST')

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode('utf-8'))
            self.last_backend = self.provider
            self.last_fallback_used = False
            return payload['choices'][0]['message']['content']
        except Exception as exc:
            self.last_error = str(exc)
            if self.fallback_client is not None:
                text = self.fallback_client.complete(prompt)
                self.last_backend = 'local_fallback_api_error'
                self.last_fallback_used = True
                return text
            raise
