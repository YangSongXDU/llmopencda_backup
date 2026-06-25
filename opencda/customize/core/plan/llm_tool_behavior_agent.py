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

    When self_perception_only is enabled, CARLA-server vehicle lists from the
    default PerceptionManager are not used by this LLM behavior path. The
    speed control decision depends on self-perceived tool outputs only.
    """

    def __init__(self, vehicle, carla_map, config_yaml):
        super(LLMToolBehaviorAgent, self).__init__(vehicle, carla_map, config_yaml)
        llm_config = config_yaml.get('llm_sensor_agent', {}) or {}
        self.self_perception_only = bool(config_yaml.get(
            'self_perception_only', True))
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

    def is_close_to_destination(self):
        """
        Disable BehaviorAgent's sys.exit-based termination for this demo.

        The original BehaviorAgent ends a scenario by calling sys.exit(0) when
        it thinks the destination is close. In this customized demo the scenario
        is already controlled by max_steps, and sys.exit would skip the logging
        loop silently. Therefore we let the scenario runner, not BehaviorAgent,
        decide when to stop.
        """
        return False

    def update_information(self, ego_pos, ego_speed, objects):
        """
        Update behavior information.

        The OpenCDA perception objects may still be produced internally for
        visualization or safety utilities. In self_perception_only mode, server
        vehicle objects are removed before entering this behavior planner.
        """
        if self.self_perception_only:
            behavior_objects = dict(objects) if isinstance(objects, dict) else {}
            behavior_objects['vehicles'] = []
            if 'traffic_lights' not in behavior_objects:
                behavior_objects['traffic_lights'] = []
        else:
            behavior_objects = objects

        super(LLMToolBehaviorAgent, self).update_information(
            ego_pos, ego_speed, behavior_objects)

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

    def destroy(self):
        self.llm_sensor_agent.destroy()
