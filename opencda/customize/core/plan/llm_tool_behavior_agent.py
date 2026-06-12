# -*- coding: utf-8 -*-
"""
BehaviorAgent wrapper that uses an LLM Sensor Tool Agent for high-level
risk-aware speed advice.
"""

from opencda.core.plan.behavior_agent import BehaviorAgent
from opencda.customize.core.llm_agent.llm_sensor_agent import LLMSensorAgent


class LLMToolBehaviorAgent(BehaviorAgent):
    """
    Keep OpenCDA's original path planning, but adjust target speed according
    to the LLM Sensor Tool Agent decision.
    """

    def __init__(self, vehicle, carla_map, config_yaml):
        super(LLMToolBehaviorAgent, self).__init__(vehicle, carla_map, config_yaml)
        llm_config = config_yaml.get('llm_sensor_agent', {}) or {}
        self.llm_sensor_agent = LLMSensorAgent(llm_config)
        self.vehicle_manager = None
        self.last_llm_decision = None
        self.last_target_speed = None
        self.last_target_location = None
        self.last_front_distance = 999.0

    def set_vehicle_manager(self, vehicle_manager):
        """
        Bind OpenCDA VehicleManager after creation.
        """
        self.vehicle_manager = vehicle_manager

    def update_information(self, ego_pos, ego_speed, objects):
        """
        Keep original BehaviorAgent updates.

        The objects argument may still be produced by OpenCDA PerceptionManager,
        but the LLM tool decision uses vehicle-mounted tool summaries instead
        of CARLA server objects for front-distance control.
        """
        super(LLMToolBehaviorAgent, self).update_information(
            ego_pos, ego_speed, objects)

        if self.vehicle_manager is not None:
            self.last_llm_decision = self.llm_sensor_agent.run_step(
                self.vehicle_manager)
            self.last_front_distance = self.last_llm_decision.front_vehicle_distance

    def run_step(self, target_speed=None, collision_detector_enabled=True,
                 lane_change_allowed=True):
        target_speed, target_location = super(LLMToolBehaviorAgent, self).run_step(
            target_speed=target_speed,
            collision_detector_enabled=collision_detector_enabled,
            lane_change_allowed=lane_change_allowed)

        if target_location is None:
            self.last_target_speed = target_speed
            self.last_target_location = target_location
            return target_speed, target_location

        if self.last_llm_decision is not None:
            target_speed = min(float(target_speed),
                               self.last_llm_decision.target_speed_advice)

        self.last_target_speed = target_speed
        self.last_target_location = target_location
        return target_speed, target_location
