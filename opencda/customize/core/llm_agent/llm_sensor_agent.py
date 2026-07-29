# -*- coding: utf-8 -*-
"""LLM Sensor Tool Agent for OpenCDA customized demos."""

import copy
import json
import time

from opencda.customize.core.tools.ego_state_tool import EgoStateTool
from opencda.customize.core.tools.camera_tool import CameraTool
from opencda.customize.core.tools.lidar_tool import LiDARTool
from opencda.customize.core.tools.radar_tool import RadarTool
from opencda.customize.core.tools.fusion_tool import FusionTool
from opencda.customize.core.tools.front_vehicle_debug_tool import \
    FrontVehicleDebugTool
from opencda.customize.core.tools.lane_check_tool import LaneCheckTool
from opencda.customize.core.llm_agent.llm_client import \
    LocalHeuristicLLMClient, RuleBasedToolSelectionClient, ChatLLMClient
from opencda.customize.core.llm_agent.prompt_builder import PromptBuilder
from opencda.customize.core.llm_agent.response_parser import \
    LLMDecision, LLMResponseParser
from opencda.customize.core.llm_agent.safety_shield import SafetyShield


def _as_list(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str) and hasattr(value, '__iter__'):
        try:
            return list(value)
        except Exception:
            pass
    return [value]


def _unique(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


class LLMSensorAgent(object):
    """
    Resource-aware LLM Agent that treats ego sensors as callable tools.

    The agent first gathers a low-cost context, asks an LLM/rule baseline which
    sensing tools are worth calling next, executes the selected tools, optionally
    performs result-level fusion, and returns a high-level driving decision.
    """

    def __init__(self, config=None):
        config = config or {}
        self.enabled = bool(config.get('enabled', True))
        self.debug = bool(config.get('debug', True))
        self.call_interval = max(1, int(config.get('llm_call_interval', 20)))
        self.legacy_initial_tools = _as_list(
            config.get('initial_tools'),
            ['camera_tool', 'lidar_tool', 'radar_tool'])

        thresholds = config.get('risk_threshold', {}) or {}
        self.medium_distance = float(thresholds.get('medium_distance', 60.0))
        self.high_distance = float(thresholds.get('high_distance', 30.0))
        self.critical_distance = float(thresholds.get('critical_distance', 12.0))

        tool_cost = config.get('tool_cost', {}) or {}
        lidar_config = config.get('lidar_tool', {}) or {}
        lidar_config['cost'] = tool_cost.get(
            'lidar_tool', lidar_config.get('cost', 2.0))
        camera_config = config.get('camera_tool', {}) or {}
        camera_config['cost'] = tool_cost.get(
            'camera_tool', camera_config.get('cost', 1.0))
        radar_config = config.get('radar_tool', {}) or {}
        radar_config['cost'] = tool_cost.get(
            'radar_tool', radar_config.get('cost', 1.5))
        fusion_config = config.get('fusion_tool', {}) or {}
        fusion_config['cost'] = tool_cost.get(
            'fusion_tool', fusion_config.get('cost', 3.0))
        front_debug_config = config.get('front_vehicle_debug_tool', {}) or {}
        front_debug_config['cost'] = tool_cost.get(
            'front_vehicle_debug_tool', front_debug_config.get('cost', 0.2))
        lane_check_config = config.get('lane_check_tool', {}) or {}
        lane_check_config['cost'] = tool_cost.get(
            'lane_check_tool', lane_check_config.get('cost', 0.5))

        self.ego_tool = EgoStateTool(cost=tool_cost.get('ego_state_tool', 0.1))
        self.camera_tool = CameraTool(camera_config)
        self.lidar_tool = LiDARTool(lidar_config)
        self.radar_tool = RadarTool(radar_config)
        self.front_vehicle_debug_tool = FrontVehicleDebugTool(front_debug_config)
        self.lane_check_tool = LaneCheckTool(lane_check_config)
        self.fusion_tool = FusionTool(fusion_config)

        self.tool_registry = {
            'camera_tool': self.camera_tool,
            'lidar_tool': self.lidar_tool,
            'radar_tool': self.radar_tool,
            'front_vehicle_debug_tool': self.front_vehicle_debug_tool,
            'lane_check_tool': self.lane_check_tool,
            'fusion_tool': self.fusion_tool
        }

        self.tool_selection_config = config.get('tool_selection', {}) or {}
        self.tool_selection_enabled = bool(
            self.tool_selection_config.get('enabled', False))
        self.tool_selection_strategy = str(
            self.tool_selection_config.get('strategy', 'llm')).lower()
        self.base_tools = _as_list(
            self.tool_selection_config.get('base_tools'),
            ['camera_tool'] if self.tool_selection_enabled
            else self.legacy_initial_tools)
        self.selectable_tools = _as_list(
            self.tool_selection_config.get('selectable_tools'),
            ['radar_tool', 'lidar_tool', 'fusion_tool'])
        self.max_tools_per_step = int(
            self.tool_selection_config.get('max_tools_per_step', 3))
        self.cost_budget_per_step = float(
            self.tool_selection_config.get('cost_budget_per_step', 6.5))
        self.allow_cached_results = bool(
            self.tool_selection_config.get('allow_cached_results', True))
        self.tool_cache_config = (
            self.tool_selection_config.get('tool_cache', {}) or {})
        self.avoid_full_fusion_when_low_risk = bool(
            self.tool_selection_config.get(
                'avoid_full_fusion_when_low_risk', True))

        prototype_config = config.get('prototype_oracle_tools', {}) or {}
        legacy_has_oracle = (
            'front_vehicle_debug_tool' in self.legacy_initial_tools or
            'lane_check_tool' in self.legacy_initial_tools)
        self.prototype_oracle_enabled = bool(
            prototype_config.get('enabled', legacy_has_oracle))
        self.prototype_oracle_tools = _as_list(
            prototype_config.get('tools'),
            ['front_vehicle_debug_tool', 'lane_check_tool']
            if self.prototype_oracle_enabled else [])
        self.prototype_oracle_counted = bool(prototype_config.get(
            'count_toward_resource_metrics', True))
        self.prototype_oracle_exposed = bool(prototype_config.get(
            'expose_to_decision', True))
        self.preflight_enabled = bool(config.get('llm_preflight_enabled', False))

        fallback_client = LocalHeuristicLLMClient(
            self.medium_distance, self.high_distance, self.critical_distance)
        provider = str(config.get('llm_provider', 'local')).lower()
        if provider in ['rule', 'rule_based'] or \
                self.tool_selection_strategy == 'rule':
            self.llm_client = RuleBasedToolSelectionClient(
                self.medium_distance, self.high_distance,
                self.critical_distance)
        elif provider in ['deepseek', 'openai', 'chat']:
            self.llm_client = ChatLLMClient(config, fallback_client=fallback_client)
        else:
            self.llm_client = fallback_client

        self.parser = LLMResponseParser()
        self.safety_shield = SafetyShield(
            self.medium_distance, self.high_distance, self.critical_distance)

        self.step = 0
        self.last_decision = None
        self.last_tool_results = {}
        self.last_called_tools = []
        self.last_resource_counted_tools = []
        self.last_requested_tools = []
        self.last_executed_tools = []
        self.last_skipped_tools = []
        self.last_cached_tools = []
        self.last_tool_runtime_ms = {}
        self.last_total_tool_runtime_ms = 0.0
        self.last_total_cost = 0.0
        self.last_oracle_tool_runtime_ms = 0.0
        self.last_oracle_tool_cost = 0.0
        self.last_agent_cycle_runtime_ms = 0.0
        self.last_tool_budget = self.cost_budget_per_step
        self.last_tool_budget_used = 0.0
        self.last_tool_budget_exceeded = False
        self.last_tool_selection_reason = ''
        self.last_uncertainty_level = 'medium'
        self.last_expected_information_gain = 'medium'
        self.last_fusion_trigger_reason = ''
        self.last_resource_budget_level = 'medium'
        self.last_safety_evidence = 'insufficient'
        self.last_oracle_tool_used = False
        self.last_oracle_results = {}
        self.last_prompt = ''
        self.last_llm_backend = getattr(self.llm_client, 'last_backend', provider)
        self.last_llm_fallback_used = False
        self.last_llm_error = ''
        self.last_llm_call_executed = False
        self.last_llm_request_latency_ms = 0.0
        self.last_llm_retry_count = 0
        self.last_llm_http_status = 0
        self.last_llm_response_valid = False
        self.last_llm_call_success = False
        self.last_llm_fallback_reason = ''
        self.last_llm_circuit_open = False
        self.tool_cache = {}

    def _reset_step_accounting(self):
        self.last_called_tools = []
        self.last_resource_counted_tools = []
        self.last_requested_tools = []
        self.last_executed_tools = []
        self.last_skipped_tools = []
        self.last_cached_tools = []
        self.last_tool_runtime_ms = {}
        self.last_total_tool_runtime_ms = 0.0
        self.last_total_cost = 0.0
        self.last_oracle_tool_runtime_ms = 0.0
        self.last_oracle_tool_cost = 0.0
        self.last_agent_cycle_runtime_ms = 0.0
        self.last_tool_budget = self.cost_budget_per_step
        self.last_tool_budget_used = 0.0
        self.last_tool_budget_exceeded = False
        self.last_tool_selection_reason = ''
        self.last_uncertainty_level = 'medium'
        self.last_expected_information_gain = 'medium'
        self.last_fusion_trigger_reason = ''
        self.last_resource_budget_level = 'medium'
        self.last_safety_evidence = 'insufficient'
        self.last_oracle_tool_used = False
        self.last_oracle_results = {}
        self.last_llm_call_executed = False
        self.last_llm_request_latency_ms = 0.0
        self.last_llm_retry_count = 0
        self.last_llm_http_status = 0
        self.last_llm_response_valid = False
        self.last_llm_call_success = False
        self.last_llm_fallback_used = False
        self.last_llm_error = ''
        self.last_llm_fallback_reason = ''
        self.last_llm_circuit_open = False

    def _tool_metadata(self):
        metadata = {
            'camera_tool': {
                'modality': 'vision',
                'cost': self.camera_tool.cost,
                'best_for': 'low-cost front ROI visual objectness cue',
                'limitations': 'does not provide reliable metric distance',
                'enabled': self.camera_tool.enabled
            },
            'lidar_tool': {
                'modality': 'geometry',
                'cost': self.lidar_tool.cost,
                'best_for': 'front obstacle distance from ego LiDAR ROI',
                'limitations': 'ROI and point-density sensitive',
                'enabled': self.lidar_tool.enabled
            },
            'radar_tool': {
                'modality': 'dynamic ranging',
                'cost': self.radar_tool.cost,
                'best_for': 'relative velocity and TTC',
                'limitations': 'may include clutter or adjacent-lane detections',
                'enabled': self.radar_tool.enabled
            },
            'fusion_tool': {
                'modality': 'result-level fusion',
                'cost': self.fusion_tool.cost,
                'best_for': 'resolving uncertainty or cross-modal conflict',
                'limitations': 'depends on upstream structured summaries',
                'enabled': self.fusion_tool.enabled
            },
            'front_vehicle_debug_tool': {
                'modality': 'prototype oracle',
                'cost': self.front_vehicle_debug_tool.cost,
                'best_for': 'CARLA actor-based front vehicle validation',
                'limitations': 'not strict ego self-perception',
                'enabled': self.front_vehicle_debug_tool.enabled
            },
            'lane_check_tool': {
                'modality': 'prototype oracle',
                'cost': self.lane_check_tool.cost,
                'best_for': 'CARLA map/actor-based lane gap validation',
                'limitations': 'not strict ego self-perception',
                'enabled': self.lane_check_tool.enabled
            }
        }
        for tool_name, item in metadata.items():
            item['tool_name'] = tool_name
        return metadata

    def _allowed_tools(self):
        return [name for name in self.selectable_tools
                if name in self.tool_registry]

    def _cache_ttl(self, tool_name):
        ttl_cfg = self.tool_cache_config.get(tool_name, 0)
        if isinstance(ttl_cfg, dict) or hasattr(ttl_cfg, 'get'):
            return int(ttl_cfg.get('ttl_steps', 0))
        try:
            return int(ttl_cfg)
        except Exception:
            return 0

    def _read_cache(self, tool_name):
        if not self.allow_cached_results:
            return None
        ttl = self._cache_ttl(tool_name)
        if ttl <= 0 or tool_name not in self.tool_cache:
            return None
        cached = self.tool_cache[tool_name]
        age = self.step - int(cached.get('step', -999999))
        if age < 0 or age > ttl:
            return None
        result = copy.deepcopy(cached.get('result', {}))
        result['from_cache'] = True
        result['fresh'] = False
        result['runtime_ms'] = 0.0
        result['cache_age_steps'] = age
        return result

    def _write_cache(self, tool_name, result):
        if not self.allow_cached_results:
            return
        if self._cache_ttl(tool_name) <= 0:
            return
        self.tool_cache[tool_name] = {
            'step': self.step,
            'result': copy.deepcopy(result)
        }

    def _previous_tool_results(self):
        previous = {}
        for tool_name, cached in self.tool_cache.items():
            age = self.step - int(cached.get('step', -999999))
            result = copy.deepcopy(cached.get('result', {}))
            result['cache_age_steps'] = age
            previous[tool_name] = result
        return previous

    def _run_tool_by_name(self, tool_name, context, tool_results,
                          allow_cache=True, account_resources=True):
        if tool_name not in self.tool_registry:
            self.last_skipped_tools.append('%s:unknown_tool' % tool_name)
            return False
        if tool_name in tool_results:
            return True

        cached = self._read_cache(tool_name) if allow_cache else None
        if cached is not None:
            tool_results[tool_name] = cached
            self.last_cached_tools.append(tool_name)
            self.last_tool_runtime_ms[tool_name] = 0.0
            return True

        tool = self.tool_registry[tool_name]
        if tool_name == 'fusion_tool':
            run_context = {
                'tool_results': tool_results,
                'fusion_trigger_reason': self.last_fusion_trigger_reason
            }
        else:
            run_context = context

        start = time.time()
        result = tool.run(run_context)
        runtime_ms = (time.time() - start) * 1000.0
        if result is None:
            self.last_skipped_tools.append('%s:no_result' % tool_name)
            return False

        result_dict = result.to_dict()
        result_dict['runtime_ms'] = runtime_ms
        result_dict['from_cache'] = False
        result_dict['fresh'] = True
        tool_results[tool_name] = result_dict
        self.last_executed_tools.append(tool_name)
        self.last_called_tools.append(tool_name)
        if account_resources:
            self.last_resource_counted_tools.append(tool_name)
            self.last_total_cost += float(result.cost)
            self.last_total_tool_runtime_ms += runtime_ms
        else:
            self.last_oracle_tool_cost += float(result.cost)
            self.last_oracle_tool_runtime_ms += runtime_ms
        self.last_tool_runtime_ms[tool_name] = runtime_ms
        self._write_cache(tool_name, result_dict)
        return True

    def _fusion_inputs_available(self, tool_results):
        count = 0
        camera = tool_results.get('camera_tool', {}) or {}
        lidar = tool_results.get('lidar_tool', {}) or {}
        radar = tool_results.get('radar_tool', {}) or {}
        if camera.get('success', False) and camera.get('image_available', False):
            count += 1
        if lidar.get('success', False):
            count += 1
        if radar.get('success', False):
            count += 1
        return count >= 2

    def _run_requested_tools(self, requested_tools, context, tool_results,
                             enforce_budget=True):
        requested_tools = _unique(requested_tools)
        non_fusion = [name for name in requested_tools if name != 'fusion_tool']
        ordered = non_fusion + (
            ['fusion_tool'] if 'fusion_tool' in requested_tools else [])

        selected_count = 0
        for tool_name in ordered:
            if tool_name not in self.tool_registry:
                self.last_skipped_tools.append('%s:unknown_tool' % tool_name)
                continue
            if tool_name in tool_results:
                continue
            if tool_name == 'fusion_tool' and \
                    not self._fusion_inputs_available(tool_results):
                self.last_skipped_tools.append(
                    'fusion_tool:insufficient_modalities')
                continue
            if enforce_budget and selected_count >= self.max_tools_per_step:
                self.last_tool_budget_exceeded = True
                self.last_skipped_tools.append('%s:max_tools_exceeded' % tool_name)
                continue

            cached = self._read_cache(tool_name)
            estimated_cost = 0.0 if cached is not None else \
                float(getattr(self.tool_registry[tool_name], 'cost', 0.0))
            if enforce_budget and \
                    self.last_tool_budget_used + estimated_cost > \
                    self.cost_budget_per_step:
                self.last_tool_budget_exceeded = True
                self.last_skipped_tools.append('%s:budget_exceeded' % tool_name)
                continue

            if cached is not None:
                tool_results[tool_name] = cached
                self.last_cached_tools.append(tool_name)
                self.last_tool_runtime_ms[tool_name] = 0.0
            else:
                self._run_tool_by_name(
                    tool_name, context, tool_results, allow_cache=False)
                self.last_tool_budget_used += estimated_cost
            selected_count += 1

    def _build_prompt(self, tool_results):
        constraints = {
            'avoid_full_fusion_when_low_risk': self.avoid_full_fusion_when_low_risk,
            'safety_first': True,
            'self_perception_only': True,
            'debug_ground_truth_may_be_present': self.prototype_oracle_enabled,
            'do_not_output_throttle_brake_steer': True,
            'allowed_maneuvers': [
                'keep_lane', 'follow_front_vehicle',
                'overtake_left', 'overtake_right', 'abort_overtake'
            ]
        }
        tool_budget = {
            'max_tools_per_step': self.max_tools_per_step,
            'cost_budget_per_step': self.cost_budget_per_step,
            'current_cost_used': self.last_tool_budget_used
        }
        return PromptBuilder.build(
            ego_state=tool_results['ego_state_tool'],
            tool_results=tool_results,
            available_tools=self._allowed_tools(),
            constraints=constraints,
            tool_metadata=self._tool_metadata(),
            previous_tool_results=self._previous_tool_results(),
            tool_budget=tool_budget)

    @staticmethod
    def _valid_json_response(text):
        if isinstance(text, dict):
            return True
        try:
            data = json.loads(LLMResponseParser._extract_json(text))
            return isinstance(data, dict)
        except Exception:
            return False

    def _sync_llm_telemetry(self):
        self.last_llm_backend = getattr(self.llm_client, 'last_backend', '')
        self.last_llm_fallback_used = bool(getattr(
            self.llm_client, 'last_fallback_used', False))
        self.last_llm_error = getattr(self.llm_client, 'last_error', '')
        self.last_llm_request_latency_ms = float(getattr(
            self.llm_client, 'last_request_latency_ms', 0.0))
        self.last_llm_retry_count = int(getattr(
            self.llm_client, 'last_retry_count', 0))
        self.last_llm_http_status = int(getattr(
            self.llm_client, 'last_http_status', 0))
        self.last_llm_call_success = bool(getattr(
            self.llm_client, 'last_call_success', True))
        self.last_llm_fallback_reason = getattr(
            self.llm_client, 'last_fallback_reason', '')
        self.last_llm_circuit_open = bool(getattr(
            self.llm_client, 'last_circuit_open', False))

    def _call_or_reuse_decision(self, tool_results):
        should_call_llm = (
            self.last_decision is None or
            self.step % self.call_interval == 0
        )

        if should_call_llm:
            prompt = self._build_prompt(tool_results)
            self.last_prompt = prompt
            llm_text = self.llm_client.complete(prompt)
            self.last_llm_call_executed = True
            self._sync_llm_telemetry()
            self.last_llm_response_valid = self._valid_json_response(llm_text)
            return self.parser.parse(
                llm_text, self.last_decision,
                allowed_tools=self._allowed_tools())

        self.last_llm_backend = 'decision_cache'
        return self.last_decision or LLMDecision(
            risk_level='low',
            reason='No previous decision; default low-risk policy.')

    def _requested_tools_from_decision(self, decision):
        requested = list(decision.tools_to_call_next or [])
        if decision.fusion_required and 'fusion_tool' not in requested:
            requested.append('fusion_tool')
        return _unique(requested)

    def _update_decision_from_fusion(self, decision, tool_results):
        fusion = tool_results.get('fusion_tool', {}) or {}
        if fusion.get('front_vehicle_detected', False):
            decision.front_vehicle_distance = float(
                fusion.get('front_vehicle_distance', 999.0))

    def _sync_decision_state(self, decision):
        self.last_tool_selection_reason = decision.tool_selection_reason
        self.last_uncertainty_level = decision.uncertainty_level
        self.last_expected_information_gain = decision.expected_information_gain
        self.last_fusion_trigger_reason = decision.fusion_trigger_reason
        self.last_resource_budget_level = decision.resource_budget_level
        self.last_safety_evidence = decision.safety_evidence

    def run_step(self, vehicle_manager):
        """Execute one resource-aware LLM Agent step."""
        cycle_start = time.time()
        self.step += 1
        self._reset_step_accounting()

        context = {
            'vehicle_manager': vehicle_manager,
            'perception_manager': vehicle_manager.perception_manager
        }

        tool_results = {}
        ego_result = self.ego_tool.run(context)
        ego_dict = ego_result.to_dict()
        tool_results['ego_state_tool'] = ego_dict
        self.last_called_tools.append('ego_state_tool')
        self.last_executed_tools.append('ego_state_tool')
        self.last_resource_counted_tools.append('ego_state_tool')
        self.last_total_cost += ego_result.cost

        if self.tool_selection_enabled:
            base_tools = self.base_tools
        else:
            base_tools = self.legacy_initial_tools

        for tool_name in base_tools:
            self._run_tool_by_name(tool_name, context, tool_results)

        if self.prototype_oracle_enabled:
            oracle_results = {}
            for tool_name in self.prototype_oracle_tools:
                self._run_tool_by_name(
                    tool_name, context, oracle_results,
                    account_resources=self.prototype_oracle_counted)
            self.last_oracle_results = oracle_results
            if self.prototype_oracle_exposed:
                tool_results.update(oracle_results)

        self.last_oracle_tool_used = bool(
            'front_vehicle_debug_tool' in self.last_oracle_results or
            'lane_check_tool' in self.last_oracle_results)

        if self.tool_selection_strategy == 'always_all':
            requested = _unique([
                name for name in self.selectable_tools
                if name != 'fusion_tool'] + ['fusion_tool'])
            self.last_requested_tools = requested
            self._run_requested_tools(
                requested, context, tool_results, enforce_budget=False)
            decision = self._call_or_reuse_decision(tool_results)
        else:
            decision = self._call_or_reuse_decision(tool_results)
            self._sync_decision_state(decision)
            requested = self._requested_tools_from_decision(decision)
            self.last_requested_tools = requested
            self._run_requested_tools(
                requested, context, tool_results,
                enforce_budget=self.tool_selection_enabled)

        if 'fusion_tool' in tool_results:
            self._update_decision_from_fusion(decision, tool_results)

        decision = self.safety_shield.apply(decision, tool_results)
        self.last_safety_evidence = getattr(
            self.safety_shield, 'last_safety_evidence', decision.safety_evidence)
        decision.safety_evidence = self.last_safety_evidence

        safety_requested = self._requested_tools_from_decision(decision)
        if 'fusion_tool' in safety_requested and \
                'fusion_tool' not in tool_results:
            self.last_requested_tools = _unique(
                self.last_requested_tools + ['fusion_tool'])
            self._run_requested_tools(
                ['fusion_tool'], context, tool_results,
                enforce_budget=self.tool_selection_enabled)
            self._update_decision_from_fusion(decision, tool_results)

        self._sync_decision_state(decision)
        self.last_decision = decision
        self.last_tool_results = tool_results
        self.last_agent_cycle_runtime_ms = (
            time.time() - cycle_start) * 1000.0

        if self.debug and self.step % self.call_interval == 0:
            print('[LLMSensorAgent] backend=%s, fallback=%s, risk=%s, maneuver=%s, requested=%s, executed=%s, cached=%s, skipped=%s, distance=%.2f, cost=%.2f, reason=%s' % (
                self.last_llm_backend,
                self.last_llm_fallback_used,
                decision.risk_level,
                decision.maneuver,
                '|'.join(self.last_requested_tools),
                '|'.join(self.last_executed_tools),
                '|'.join(self.last_cached_tools),
                '|'.join(self.last_skipped_tools),
                decision.front_vehicle_distance,
                self.last_total_cost,
                decision.reason))

        return decision

    def preflight(self):
        """Check the configured LLM backend before a long simulation run."""
        if hasattr(self.llm_client, 'health_check'):
            return self.llm_client.health_check()
        return {
            'provider': getattr(self.llm_client, 'backend_name', 'local'),
            'model': '',
            'provider_available': True,
            'fallback_used': False,
            'http_status': 0,
            'latency_ms': 0.0,
            'retry_count': 0,
            'error': ''
        }

    def destroy(self):
        self.radar_tool.destroy()
