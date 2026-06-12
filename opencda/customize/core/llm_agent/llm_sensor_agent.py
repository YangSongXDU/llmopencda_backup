# -*- coding: utf-8 -*-
"""LLM Sensor Tool Agent for OpenCDA customized demos."""

from opencda.customize.core.tools.ego_state_tool import EgoStateTool
from opencda.customize.core.tools.camera_tool import CameraTool
from opencda.customize.core.tools.lidar_tool import LiDARTool
from opencda.customize.core.tools.fusion_tool import FusionTool
from opencda.customize.core.llm_agent.llm_client import LocalHeuristicLLMClient
from opencda.customize.core.llm_agent.prompt_builder import PromptBuilder
from opencda.customize.core.llm_agent.response_parser import LLMResponseParser
from opencda.customize.core.llm_agent.safety_shield import SafetyShield


class LLMSensorAgent(object):
    """
    High-level LLM Agent that calls sensor tools and decides risk/advice.

    The first runnable version uses LocalHeuristicLLMClient as an offline LLM
    substitute. The interface is designed so a real LLM client can replace it
    later without changing the planning agent.
    """

    def __init__(self, config=None):
        config = config or {}
        self.enabled = bool(config.get('enabled', True))
        self.debug = bool(config.get('debug', True))
        self.call_interval = int(config.get('llm_call_interval', 20))

        thresholds = config.get('risk_threshold', {}) or {}
        self.medium_distance = float(thresholds.get('medium_distance', 60.0))
        self.high_distance = float(thresholds.get('high_distance', 30.0))
        self.critical_distance = float(thresholds.get('critical_distance', 12.0))

        tool_cost = config.get('tool_cost', {}) or {}
        lidar_config = config.get('lidar_tool', {}) or {}
        lidar_config['cost'] = tool_cost.get('lidar_tool', lidar_config.get('cost', 2.0))
        camera_config = config.get('camera_tool', {}) or {}
        camera_config['cost'] = tool_cost.get('camera_tool', camera_config.get('cost', 1.0))
        fusion_config = config.get('fusion_tool', {}) or {}
        fusion_config['cost'] = tool_cost.get('fusion_tool', fusion_config.get('cost', 3.0))

        self.ego_tool = EgoStateTool(cost=tool_cost.get('ego_state_tool', 0.1))
        self.camera_tool = CameraTool(camera_config)
        self.lidar_tool = LiDARTool(lidar_config)
        self.fusion_tool = FusionTool(fusion_config)

        self.llm_client = LocalHeuristicLLMClient(
            self.medium_distance, self.high_distance, self.critical_distance)
        self.parser = LLMResponseParser()
        self.safety_shield = SafetyShield(
            self.medium_distance, self.high_distance, self.critical_distance)

        self.step = 0
        self.last_decision = None
        self.last_tool_results = {}
        self.last_called_tools = []
        self.last_total_cost = 0.0
        self.last_prompt = ''

    def run_step(self, vehicle_manager):
        """
        Execute one LLM Agent perception-decision step.
        """
        self.step += 1

        context = {
            'vehicle_manager': vehicle_manager,
            'perception_manager': vehicle_manager.perception_manager
        }

        tool_results = {}
        called_tools = []
        total_cost = 0.0

        ego_result = self.ego_tool.run(context)
        tool_results['ego_state_tool'] = ego_result.to_dict()
        called_tools.append('ego_state_tool')
        total_cost += ego_result.cost

        # Low-cost first-stage observation.
        camera_result = self.camera_tool.run(context)
        tool_results['camera_tool'] = camera_result.to_dict()
        called_tools.append('camera_tool')
        total_cost += camera_result.cost

        # LiDAR is the first real autonomous front-distance tool in this demo.
        lidar_result = self.lidar_tool.run(context)
        tool_results['lidar_tool'] = lidar_result.to_dict()
        called_tools.append('lidar_tool')
        total_cost += lidar_result.cost

        should_call_llm = (
            self.last_decision is None or
            self.step % self.call_interval == 0
        )

        if should_call_llm:
            prompt = PromptBuilder.build(
                ego_state=tool_results['ego_state_tool'],
                tool_results=tool_results,
                available_tools=['camera_tool', 'lidar_tool', 'fusion_tool'],
                constraints={
                    'avoid_full_fusion_when_low_risk': True,
                    'safety_first': True,
                    'do_not_output_throttle_brake_steer': True
                })
            self.last_prompt = prompt
            llm_text = self.llm_client.complete(prompt)
            decision = self.parser.parse(llm_text, self.last_decision)
        else:
            decision = self.last_decision

        decision = self.safety_shield.apply(decision, tool_results)

        if decision.fusion_required or 'fusion_tool' in decision.tools_to_call_next:
            fusion_context = {'tool_results': tool_results}
            fusion_result = self.fusion_tool.run(fusion_context)
            tool_results['fusion_tool'] = fusion_result.to_dict()
            called_tools.append('fusion_tool')
            total_cost += fusion_result.cost

            fusion_distance = fusion_result.data.get('front_vehicle_distance', 999.0)
            if fusion_result.data.get('front_vehicle_detected', False):
                decision.front_vehicle_distance = float(fusion_distance)

        self.last_decision = decision
        self.last_tool_results = tool_results
        self.last_called_tools = called_tools
        self.last_total_cost = total_cost

        if self.debug and self.step % self.call_interval == 0:
            print('[LLMSensorAgent] risk=%s, tools=%s, distance=%.2f, advice=%s, cost=%.2f, reason=%s' % (
                decision.risk_level,
                '|'.join(called_tools),
                decision.front_vehicle_distance,
                decision.driving_advice,
                total_cost,
                decision.reason))

        return decision
