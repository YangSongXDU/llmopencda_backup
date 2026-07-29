# -*- coding: utf-8 -*-
"""
LLM client abstraction for the OpenCDA LLM Agent demo.

The default offline mode is deterministic so the demo can run without an API
key. A DeepSeek/OpenAI-compatible chat client is also provided for real LLM
calls when an API key is configured.
"""

import json
import os
import time
import urllib.error
import urllib.request


class LocalHeuristicLLMClient(object):
    """Offline deterministic stand-in for an LLM with tool selection."""

    def __init__(self, medium_distance=60.0, high_distance=30.0,
                 critical_distance=12.0, backend_name='local'):
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
        self.backend_name = backend_name
        self.last_backend = backend_name
        self.last_fallback_used = False
        self.last_error = ''
        self.last_request_latency_ms = 0.0
        self.last_retry_count = 0
        self.last_http_status = 0
        self.last_response_valid = True
        self.last_call_success = True
        self.last_fallback_reason = ''
        self.last_circuit_open = False

    def _decision(self, risk, distance, advice, speed, fusion, reason,
                  maneuver='keep_lane', target_lane='current',
                  lane_change_required=False, tools=None,
                  tool_selection_reason='', uncertainty_level='medium',
                  expected_information_gain='medium',
                  fusion_trigger_reason='', resource_budget_level='medium'):
        selected_tools = tools
        if selected_tools is None:
            selected_tools = ['fusion_tool'] if fusion else []
        return {
            'tools_to_call_next': selected_tools,
            'fusion_required': bool(fusion),
            'tool_selection_reason': tool_selection_reason,
            'uncertainty_level': uncertainty_level,
            'expected_information_gain': expected_information_gain,
            'fusion_trigger_reason': fusion_trigger_reason,
            'resource_budget_level': resource_budget_level,
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
        self.last_backend = self.backend_name
        self.last_fallback_used = False
        self.last_error = ''
        self.last_request_latency_ms = 0.0
        self.last_retry_count = 0
        self.last_http_status = 0
        self.last_response_valid = True
        self.last_call_success = True
        self.last_fallback_reason = ''
        self.last_circuit_open = False

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
        left_clear = bool(lane.get(
            'left_overtake_suitable',
            lane.get('left_lane_exists', False) and
            lane.get('left_lane_clear', False)))
        right_clear = bool(lane.get(
            'right_overtake_suitable',
            lane.get('right_lane_exists', False) and
            lane.get('right_lane_clear', False)))

        # First overtaking prototype: use the debug front-vehicle and lane-check
        # tools when enabled. They make the maneuver demo deterministic and also
        # expose why ego may see distance=999 from pure sensor tools.
        if debug_front_detected and debug_front_distance < self.overtake_distance:
            if left_clear:
                return json.dumps(self._decision(
                    'medium', debug_front_distance, 'keep_speed', 40.0, True,
                    'A slower front vehicle is present and the left lane is clear; request overtake_left.',
                    maneuver='overtake_left', target_lane='left',
                    lane_change_required=True,
                    tools=['radar_tool', 'lidar_tool', 'fusion_tool'],
                    tool_selection_reason='Front debug cue and clear left lane justify radar/LiDAR confirmation before overtaking.',
                    uncertainty_level='medium',
                    expected_information_gain='high',
                    fusion_trigger_reason='Overtaking request should be confirmed with ranging modalities.',
                    resource_budget_level='high'))
            if right_clear:
                return json.dumps(self._decision(
                    'medium', debug_front_distance, 'keep_speed', 35.0, True,
                    'A slower front vehicle is present and the right lane is clear; request overtake_right.',
                    maneuver='overtake_right', target_lane='right',
                    lane_change_required=True,
                    tools=['radar_tool', 'lidar_tool', 'fusion_tool'],
                    tool_selection_reason='Front debug cue and clear right lane justify radar/LiDAR confirmation before overtaking.',
                    uncertainty_level='medium',
                    expected_information_gain='high',
                    fusion_trigger_reason='Overtaking request should be confirmed with ranging modalities.',
                    resource_budget_level='high'))
            return json.dumps(self._decision(
                'high', debug_front_distance, 'slow_down', 18.0, True,
                'A front vehicle is present but no adjacent lane is clear; follow the front vehicle.',
                maneuver='follow_front_vehicle', target_lane='current',
                lane_change_required=False,
                tools=['radar_tool', 'lidar_tool', 'fusion_tool'],
                tool_selection_reason='Blocked front vehicle with no safe lane requires stronger ranging evidence.',
                uncertainty_level='medium',
                expected_information_gain='high',
                fusion_trigger_reason='High-risk following should use result-level confirmation.',
                resource_budget_level='high'))

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
                maneuver='follow_front_vehicle',
                tools=['radar_tool', 'fusion_tool'],
                tool_selection_reason='Critical LiDAR distance asks radar/fusion confirmation while slowing.',
                uncertainty_level='high',
                expected_information_gain='high',
                fusion_trigger_reason='Critical distance.',
                resource_budget_level='high'))

        if radar_reliable_ttc and radar_ttc < self.critical_ttc:
            return json.dumps(self._decision(
                'critical', radar_distance, 'emergency_slow', 5.0, True,
                'Radar self-perception reports a critical TTC.',
                maneuver='follow_front_vehicle',
                tools=['lidar_tool', 'fusion_tool'],
                tool_selection_reason='Critical radar TTC asks LiDAR/fusion confirmation while slowing.',
                uncertainty_level='high',
                expected_information_gain='high',
                fusion_trigger_reason='Critical TTC.',
                resource_budget_level='high'))

        if lidar_detected and lidar_distance < self.high_distance:
            return json.dumps(self._decision(
                'high', lidar_distance, 'slow_down', 15.0, True,
                'LiDAR self-perception reports a high-risk front distance.',
                maneuver='follow_front_vehicle',
                tools=['radar_tool', 'fusion_tool'],
                tool_selection_reason='High LiDAR risk asks radar confirmation.',
                uncertainty_level='medium',
                expected_information_gain='high',
                fusion_trigger_reason='High front-distance risk.',
                resource_budget_level='high'))

        if radar_reliable_ttc and radar_ttc < self.high_ttc:
            return json.dumps(self._decision(
                'high', radar_distance, 'slow_down', 15.0, True,
                'Radar self-perception reports a high-risk TTC.',
                maneuver='follow_front_vehicle',
                tools=['lidar_tool', 'fusion_tool'],
                tool_selection_reason='High radar TTC asks LiDAR confirmation.',
                uncertainty_level='medium',
                expected_information_gain='high',
                fusion_trigger_reason='High TTC risk.',
                resource_budget_level='high'))

        if lidar_detected and lidar_distance < self.medium_distance:
            return json.dumps(self._decision(
                'medium', lidar_distance, 'slow_down', 30.0, True,
                'LiDAR self-perception reports a medium-range front object.',
                maneuver='follow_front_vehicle',
                tools=['radar_tool', 'fusion_tool'],
                tool_selection_reason='Medium LiDAR distance asks radar if budget allows.',
                uncertainty_level='medium',
                expected_information_gain='medium',
                fusion_trigger_reason='Medium-range object confirmation.',
                resource_budget_level='medium'))

        weak_multimodal_cue = (
            camera_possible and camera_conf >= 0.45 and
            (lidar_detected or (radar_detected and radar_conf >= 0.50))
        )
        if weak_multimodal_cue:
            return json.dumps(self._decision(
                'low', distance, 'keep_speed', 50.0, True,
                'Weak multimodal cue exists; call fusion for confirmation.',
                tools=['fusion_tool'],
                tool_selection_reason='Weak multimodal cue is enough to justify lightweight fusion only.',
                uncertainty_level='medium',
                expected_information_gain='medium',
                fusion_trigger_reason='Weak but consistent multimodal cue.',
                resource_budget_level='medium'))

        return json.dumps(self._decision(
            'low', 999.0, 'keep_speed', 50.0, False,
            'No reliable close front object is detected by ego sensor tools.',
            tools=[],
            tool_selection_reason='Low risk: keep the minimum tool set.',
            uncertainty_level='low',
            expected_information_gain='low',
            fusion_trigger_reason='',
            resource_budget_level='low'))


class RuleBasedToolSelectionClient(LocalHeuristicLLMClient):
    """Deterministic rule-based baseline for selective sensor tools."""

    def __init__(self, medium_distance=60.0, high_distance=30.0,
                 critical_distance=12.0):
        super(RuleBasedToolSelectionClient, self).__init__(
            medium_distance, high_distance, critical_distance,
            backend_name='rule_based')


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
        self.mode = str(config.get('llm_mode', 'robust_demo')).lower()
        self.allow_fallback = bool(config.get(
            'llm_allow_fallback', self.mode != 'paper_llm_only'))
        self.max_retries = max(0, int(config.get('llm_max_retries', 2)))
        self.retry_backoff = max(
            0.0, float(config.get('llm_retry_backoff', 0.5)))
        self.circuit_failure_threshold = max(
            0, int(config.get('llm_circuit_breaker_failures', 3)))
        self.circuit_cooldown_calls = max(
            0, int(config.get('llm_circuit_breaker_cooldown_calls', 3)))
        self.fallback_client = fallback_client
        self.last_backend = self.provider
        self.last_fallback_used = False
        self.last_error = ''
        self.last_request_latency_ms = 0.0
        self.last_retry_count = 0
        self.last_http_status = 0
        self.last_response_valid = False
        self.last_call_success = False
        self.last_fallback_reason = ''
        self.last_circuit_open = False
        self._consecutive_failures = 0
        self._circuit_calls_remaining = 0

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

    def _reset_telemetry(self):
        self.last_backend = self.provider
        self.last_fallback_used = False
        self.last_error = ''
        self.last_request_latency_ms = 0.0
        self.last_retry_count = 0
        self.last_http_status = 0
        self.last_response_valid = False
        self.last_call_success = False
        self.last_fallback_reason = ''
        self.last_circuit_open = False

    @staticmethod
    def _http_status(exc):
        try:
            return int(getattr(exc, 'code', 0) or 0)
        except Exception:
            return 0

    @classmethod
    def _retryable(cls, exc):
        status = cls._http_status(exc)
        if status:
            return status == 429 or status >= 500
        return isinstance(exc, (
            urllib.error.URLError, TimeoutError, ConnectionError))

    def _fallback(self, prompt, backend, reason):
        self.last_fallback_reason = reason
        if not self.allow_fallback or self.fallback_client is None:
            raise RuntimeError(self.last_error or reason)
        text = self.fallback_client.complete(prompt)
        self.last_backend = backend
        self.last_fallback_used = True
        return text

    def complete(self, prompt):
        self._reset_telemetry()
        request_start = time.time()

        api_key = os.environ.get(self.api_key_env, '')
        if not api_key:
            self.last_error = 'Missing API key environment variable: %s' % self.api_key_env
            self.last_request_latency_ms = (time.time() - request_start) * 1000.0
            return self._fallback(
                prompt, 'local_fallback_missing_key', 'missing_api_key')

        if self._circuit_calls_remaining > 0:
            self._circuit_calls_remaining -= 1
            self.last_circuit_open = True
            self.last_error = (
                'LLM circuit breaker open; %d skipped calls remain' %
                self._circuit_calls_remaining)
            self.last_request_latency_ms = (time.time() - request_start) * 1000.0
            return self._fallback(
                prompt, 'local_fallback_circuit_open', 'circuit_open')

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

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    self.last_http_status = int(
                        getattr(resp, 'status', 0) or resp.getcode() or 0)
                    payload = json.loads(resp.read().decode('utf-8'))
                content = payload['choices'][0]['message']['content']
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('LLM response content is empty')
                self.last_backend = self.provider
                self.last_fallback_used = False
                self.last_response_valid = True
                self.last_call_success = True
                self.last_retry_count = attempt
                self.last_request_latency_ms = (
                    time.time() - request_start) * 1000.0
                self._consecutive_failures = 0
                return content
            except Exception as exc:
                last_exc = exc
                self.last_error = str(exc)
                self.last_http_status = self._http_status(exc)
                self.last_retry_count = attempt
                if attempt < self.max_retries and self._retryable(exc):
                    if self.retry_backoff > 0.0:
                        time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                break

        self.last_request_latency_ms = (time.time() - request_start) * 1000.0
        self._consecutive_failures += 1
        if self.circuit_failure_threshold > 0 and \
                self._consecutive_failures >= self.circuit_failure_threshold:
            self._circuit_calls_remaining = self.circuit_cooldown_calls
            self.last_circuit_open = self._circuit_calls_remaining > 0
        reason = 'api_error'
        if last_exc is not None and self.last_http_status:
            reason = 'http_%d' % self.last_http_status
        elif last_exc is not None and self._retryable(last_exc):
            reason = 'network_or_timeout'
        return self._fallback(
            prompt, 'local_fallback_api_error', reason)

    def health_check(self):
        """Issue a minimal request and report whether the configured API answered."""
        prompt = json.dumps({
            'task': 'Return a valid low-risk driving decision JSON.',
            'required': {
                'tools_to_call_next': [],
                'fusion_required': False,
                'risk_level': 'low',
                'front_vehicle_distance': 999.0,
                'driving_advice': 'keep_speed',
                'target_speed_advice': 20.0,
                'maneuver': 'keep_lane',
                'target_lane': 'current',
                'lane_change_required': False,
                'reason': 'preflight'
            }
        })
        error = ''
        try:
            self.complete(prompt)
        except Exception as exc:
            error = str(exc)
        return {
            'provider': self.provider,
            'model': self.model,
            'provider_available': bool(self.last_call_success),
            'fallback_used': bool(self.last_fallback_used),
            'http_status': int(self.last_http_status),
            'latency_ms': float(self.last_request_latency_ms),
            'retry_count': int(self.last_retry_count),
            'error': error or self.last_error
        }
