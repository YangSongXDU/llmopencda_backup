# -*- coding: utf-8 -*-
"""
BehaviorAgent wrapper that uses an LLM Sensor Tool Agent for high-level
risk-aware speed and maneuver advice.
"""

import carla

from opencda.core.plan.behavior_agent import BehaviorAgent
from opencda.customize.core.llm_agent.llm_sensor_agent import LLMSensorAgent


class LLMToolBehaviorAgent(BehaviorAgent):
    """
    Keep OpenCDA's original path planning and controller, but allow the LLM
    Sensor Agent to limit speed and request high-level maneuvers such as a
    prototype left/right overtake.
    """

    def __init__(self, vehicle, carla_map, config_yaml):
        super(LLMToolBehaviorAgent, self).__init__(vehicle, carla_map, config_yaml)
        llm_config = config_yaml.get('llm_sensor_agent', {}) or {}
        overtake_config = config_yaml.get('llm_overtake', {}) or {}
        self.self_perception_only = bool(config_yaml.get(
            'self_perception_only', True))
        self.llm_sensor_agent = LLMSensorAgent(llm_config)
        self.vehicle_manager = None
        self.last_llm_decision = None
        self.last_target_speed = None
        self.last_target_location = None
        self.last_front_distance = 999.0
        self.overtake_enabled = bool(overtake_config.get('enabled', True))
        self.overtake_lookahead = float(overtake_config.get('lookahead_distance', 45.0))
        self.overtake_cooldown_steps = int(overtake_config.get('cooldown_steps', 80))
        self.overtake_cooldown = 0
        self.last_maneuver_applied = 'none'
        self.last_maneuver_reason = ''

    def set_vehicle_manager(self, vehicle_manager):
        """Bind OpenCDA VehicleManager after creation."""
        self.vehicle_manager = vehicle_manager

    def is_close_to_destination(self):
        """
        Disable BehaviorAgent's sys.exit-based termination for this demo.
        """
        return False

    def update_information(self, ego_pos, ego_speed, objects):
        """
        Update behavior information.

        In self_perception_only mode, CARLA-server vehicle objects are removed
        before entering the default BehaviorAgent collision path. The LLM tools
        can still run separately through vehicle_manager context.
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

    def _lane_check_data(self):
        tool_results = getattr(self.llm_sensor_agent, 'last_tool_results', {}) or {}
        lane = tool_results.get('lane_check_tool', {}) or {}
        nested = lane.get('data', None)
        if isinstance(nested, dict):
            lane = nested
        return lane

    def _adjacent_lane_waypoint(self, current_wp, maneuver):
        if maneuver == 'overtake_left':
            target_wp = current_wp.get_left_lane()
        elif maneuver == 'overtake_right':
            target_wp = current_wp.get_right_lane()
        else:
            return None

        if target_wp is None:
            return None
        if target_wp.lane_type != carla.LaneType.Driving:
            return None
        if target_wp.lane_id * current_wp.lane_id <= 0:
            return None
        return target_wp

    def _apply_llm_maneuver_if_needed(self):
        """Convert an LLM overtake maneuver into a temporary lane-change route."""
        self.last_maneuver_applied = 'none'
        self.last_maneuver_reason = ''

        if self.overtake_cooldown > 0:
            self.overtake_cooldown -= 1
            self.last_maneuver_reason = 'overtake cooldown active'
            return

        decision = self.last_llm_decision
        if decision is None or not self.overtake_enabled:
            return
        if decision.maneuver not in ['overtake_left', 'overtake_right']:
            return

        lane = self._lane_check_data()
        if decision.maneuver == 'overtake_left':
            lane_clear = bool(lane.get('left_lane_exists', False) and
                              lane.get('left_lane_clear', False))
        else:
            lane_clear = bool(lane.get('right_lane_exists', False) and
                              lane.get('right_lane_clear', False))
        if not lane_clear:
            self.last_maneuver_reason = 'target adjacent lane is not clear'
            return

        current_wp = self._map.get_waypoint(self._ego_pos.location)
        target_wp = self._adjacent_lane_waypoint(current_wp, decision.maneuver)
        if target_wp is None:
            self.last_maneuver_reason = 'target adjacent lane does not exist'
            return

        next_wps = target_wp.next(self.overtake_lookahead)
        lane_target = next_wps[0] if next_wps else target_wp
        self.set_destination(
            self._ego_pos.location,
            lane_target.transform.location,
            clean=True,
            end_reset=False)
        self.overtake_cooldown = self.overtake_cooldown_steps
        self.last_maneuver_applied = decision.maneuver
        self.last_maneuver_reason = 'temporary lane-change route injected'
        if self.debug:
            print('[LLMToolBehaviorAgent] maneuver=%s target=(%.2f, %.2f)' % (
                decision.maneuver,
                lane_target.transform.location.x,
                lane_target.transform.location.y))

    def run_step(self, target_speed=None, collision_detector_enabled=True,
                 lane_change_allowed=True):
        self._apply_llm_maneuver_if_needed()

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
