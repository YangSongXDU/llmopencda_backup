# -*- coding: utf-8 -*-
"""
BehaviorAgent wrapper that uses an LLM Sensor Tool Agent for high-level
risk-aware speed and maneuver advice.
"""

import math

import carla

from opencda.core.plan.behavior_agent import BehaviorAgent
from opencda.customize.core.llm_agent.llm_sensor_agent import LLMSensorAgent


class LLMToolBehaviorAgent(BehaviorAgent):
    """
    Keep OpenCDA's original path planning and controller, but allow the LLM
    Sensor Agent to limit speed and request high-level maneuvers.

    This wrapper now contains a small overtaking state machine:
      KEEP_LANE -> CHANGING_LANE -> PASSING -> RETURNING -> OVERTAKE_DONE
                        |             |
                        +-> ABORTING <-+
                <---------------- cooldown ------------------|
    The state machine converts LLM maneuvers into temporary OpenCDA routes and
    can repeat the cycle for successive slower vehicles.
    """

    KEEP_LANE = 'KEEP_LANE'
    CHANGING_LANE = 'CHANGING_LANE'
    PASSING = 'PASSING'
    RETURNING = 'RETURNING'
    ABORTING = 'ABORTING'
    OVERTAKE_DONE = 'OVERTAKE_DONE'

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
        self.return_lookahead = float(overtake_config.get('return_lookahead_distance', 45.0))
        self.lane_route_entry_distance = float(
            overtake_config.get('lane_route_entry_distance', 5.0))
        self.pass_distance_margin = float(overtake_config.get('pass_distance_margin', 12.0))
        self.min_passing_distance = float(overtake_config.get('min_passing_distance', 65.0))
        self.overtake_cooldown_steps = int(overtake_config.get('cooldown_steps', 80))
        self.max_overtakes = int(overtake_config.get('max_overtakes', 1))
        self.max_overtake_attempts = int(
            overtake_config.get('max_overtake_attempts', 0))
        self.min_target_lane_continuity = float(
            overtake_config.get('min_target_lane_continuity', 0.0))
        self.min_remaining_lane_continuity = float(
            overtake_config.get('min_remaining_lane_continuity', 0.0))
        self.max_lane_change_steps = int(
            overtake_config.get('max_lane_change_steps', 180))
        self.max_passing_steps = int(
            overtake_config.get('max_passing_steps', 0))
        self.max_passing_distance = float(
            overtake_config.get('max_passing_distance', 0.0))
        self.overtake_cooldown = 0
        self.strict_original_road_match = bool(
            overtake_config.get('strict_original_road_match', True))

        self.overtake_state = self.KEEP_LANE
        self.original_lane_id = 0
        self.original_road_id = 0
        self.target_lane_id = 0
        self.tracked_front_actor_id = -1
        self.overtake_start_location = None
        self.passed_front_vehicle = False
        self.return_lane_clear = False
        self.overtake_attempt_count = 0
        self.completed_overtake_count = 0
        self.overtake_abort_count = 0
        self.lane_change_steps = 0
        self.passing_steps = 0
        self.target_lane_continuous_distance = 999.0
        self.target_lane_ends_or_merges_ahead = False
        self.last_overtake_abort_reason = ''
        self.abort_route_injected = False
        self.last_maneuver_applied = 'none'
        self.last_maneuver_reason = ''

    def set_vehicle_manager(self, vehicle_manager):
        """Bind OpenCDA VehicleManager after creation."""
        self.vehicle_manager = vehicle_manager

    def is_close_to_destination(self):
        """Disable BehaviorAgent's sys.exit-based termination for this demo."""
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

    def _tool_data(self, tool_name):
        tool_results = getattr(self.llm_sensor_agent, 'last_tool_results', {}) or {}
        data = tool_results.get(tool_name, {}) or {}
        nested = data.get('data', None)
        if isinstance(nested, dict):
            merged = dict(data)
            merged.update(nested)
            return merged
        return data

    def _lane_check_data(self):
        return self._tool_data('lane_check_tool')

    def _front_debug_data(self):
        return self._tool_data('front_vehicle_debug_tool')

    def _adjacent_lane_waypoint(self, current_wp, maneuver):
        if maneuver == 'overtake_left':
            target_wp = current_wp.get_left_lane()
        elif maneuver == 'overtake_right':
            target_wp = current_wp.get_right_lane()
        else:
            return None
        return self._validate_adjacent_lane(current_wp, target_wp)

    @staticmethod
    def _validate_adjacent_lane(current_wp, target_wp):
        if target_wp is None:
            return None
        if target_wp.lane_type != carla.LaneType.Driving:
            return None
        if target_wp.lane_id * current_wp.lane_id <= 0:
            return None
        return target_wp

    def _matches_original_lane(self, waypoint):
        if waypoint.lane_id != self.original_lane_id:
            return False
        if self.strict_original_road_match and \
                waypoint.road_id != self.original_road_id:
            return False
        return True

    def _find_original_lane_waypoint(self, current_wp):
        """Find the waypoint of the original lane adjacent to current lane."""
        if self._matches_original_lane(current_wp):
            return current_wp

        for candidate in [current_wp.get_left_lane(), current_wp.get_right_lane()]:
            candidate = self._validate_adjacent_lane(current_wp, candidate)
            if candidate is None:
                continue
            if self._matches_original_lane(candidate):
                return candidate
        return None

    def _current_lane_id(self):
        try:
            return int(self._map.get_waypoint(self._ego_pos.location).lane_id)
        except Exception:
            return 0

    def _is_back_on_original_lane(self):
        current_wp = self._map.get_waypoint(self._ego_pos.location)
        return self._matches_original_lane(current_wp)

    def _lane_clear_for_original_return(self, current_wp):
        if self._is_back_on_original_lane():
            return True

        lane = self._lane_check_data()
        left_matches = bool(
            lane.get('left_lane_exists', False) and
            int(lane.get('left_lane_id', 0)) == self.original_lane_id)
        right_matches = bool(
            lane.get('right_lane_exists', False) and
            int(lane.get('right_lane_id', 0)) == self.original_lane_id)

        if left_matches:
            return bool(lane.get('left_lane_clear', False))
        if right_matches:
            return bool(lane.get('right_lane_clear', False))
        return False

    @staticmethod
    def _longitudinal_distance(ego_transform, target_location):
        yaw = math.radians(ego_transform.rotation.yaw)
        ego_loc = ego_transform.location
        dx = target_location.x - ego_loc.x
        dy = target_location.y - ego_loc.y
        return dx * math.cos(yaw) + dy * math.sin(yaw)

    def _distance_from_overtake_start(self):
        if self.overtake_start_location is None:
            return 0.0
        loc = self._ego_pos.location
        return math.sqrt(
            (loc.x - self.overtake_start_location.x) ** 2 +
            (loc.y - self.overtake_start_location.y) ** 2)

    def _has_passed_tracked_front_vehicle(self):
        """Return True once the tracked front vehicle is safely behind ego."""
        if self.tracked_front_actor_id > 0:
            try:
                actor = self.vehicle.get_world().get_actor(self.tracked_front_actor_id)
                if actor is not None and actor.is_alive:
                    longitudinal = self._longitudinal_distance(
                        self.vehicle.get_transform(), actor.get_location())
                    return longitudinal < -self.pass_distance_margin
            except Exception:
                pass

        # Fallback for cases where the debug actor disappeared: after ego has
        # driven far enough in the passing lane, allow return evaluation.
        return self._distance_from_overtake_start() > self.min_passing_distance

    def _inject_route_to_lane(self, lane_wp, lookahead, applied, reason):
        if lane_wp is None:
            self.last_maneuver_reason = 'target lane waypoint missing'
            return False
        entry_wps = lane_wp.next(self.lane_route_entry_distance)
        entry_wp = entry_wps[0] if entry_wps else lane_wp
        next_wps = lane_wp.next(lookahead)
        target_wp = next_wps[0] if next_wps else lane_wp
        self.set_destination(
            entry_wp.transform.location,
            target_wp.transform.location,
            clean=True,
            end_reset=False)
        self.last_maneuver_applied = applied
        self.last_maneuver_reason = reason
        if self.debug:
            print('[LLMToolBehaviorAgent] state=%s applied=%s target=(%.2f, %.2f)' % (
                self.overtake_state,
                applied,
                target_wp.transform.location.x,
                target_wp.transform.location.y))
        return True

    def _restore_original_destination(self):
        if self.end_waypoint is None:
            return
        try:
            self.set_destination(
                self._ego_pos.location,
                self.end_waypoint.transform.location,
                clean=True,
                end_reset=False)
        except Exception:
            pass

    def _reset_overtake_context(self):
        """Clear per-cycle state before looking for the next front vehicle."""
        self.original_lane_id = 0
        self.original_road_id = 0
        self.target_lane_id = 0
        self.tracked_front_actor_id = -1
        self.overtake_start_location = None
        self.passed_front_vehicle = False
        self.return_lane_clear = False
        self.lane_change_steps = 0
        self.passing_steps = 0
        self.target_lane_continuous_distance = 999.0
        self.target_lane_ends_or_merges_ahead = False
        self.abort_route_injected = False

    def _begin_abort(self, reason):
        """Stop the current attempt and route back toward the original lane."""
        if self.overtake_state == self.ABORTING:
            return

        self.overtake_state = self.ABORTING
        self.overtake_abort_count += 1
        self.last_overtake_abort_reason = reason
        self.last_maneuver_applied = 'abort_overtake'
        self.last_maneuver_reason = reason
        self.abort_route_injected = False

        current_wp = self._map.get_waypoint(self._ego_pos.location)
        if self._matches_original_lane(current_wp):
            self._restore_original_destination()
            return

        if self._lane_clear_for_original_return(current_wp):
            original_wp = self._find_original_lane_waypoint(current_wp)
            if original_wp is not None:
                self.abort_route_injected = self._inject_route_to_lane(
                    original_wp,
                    self.return_lookahead,
                    'abort_overtake',
                    reason)
                return

        self._restore_original_destination()

    def _start_overtake_if_requested(self):
        decision = self.last_llm_decision
        if decision is None or not self.overtake_enabled:
            return
        if decision.maneuver not in ['overtake_left', 'overtake_right']:
            return
        if self.max_overtake_attempts > 0 and \
                self.overtake_attempt_count >= self.max_overtake_attempts:
            self.last_maneuver_reason = (
                'configured maximum number of overtaking attempts reached')
            return

        lane = self._lane_check_data()
        if decision.maneuver == 'overtake_left':
            lane_clear = bool(lane.get('left_lane_exists', False) and
                              lane.get('left_lane_clear', False))
            continuity = float(
                lane.get('left_lane_continuous_distance', 999.0))
            ends_or_merges = bool(
                lane.get('left_lane_ends_or_merges_ahead', False))
        else:
            lane_clear = bool(lane.get('right_lane_exists', False) and
                              lane.get('right_lane_clear', False))
            continuity = float(
                lane.get('right_lane_continuous_distance', 999.0))
            ends_or_merges = bool(
                lane.get('right_lane_ends_or_merges_ahead', False))
        self.target_lane_continuous_distance = continuity
        self.target_lane_ends_or_merges_ahead = ends_or_merges
        if not lane_clear:
            self.last_maneuver_reason = 'target adjacent lane is not clear'
            return
        if continuity < self.min_target_lane_continuity:
            self.last_maneuver_reason = (
                'target lane continuity %.1fm is below required %.1fm' % (
                    continuity, self.min_target_lane_continuity))
            return

        current_wp = self._map.get_waypoint(self._ego_pos.location)
        target_wp = self._adjacent_lane_waypoint(current_wp, decision.maneuver)
        if target_wp is None:
            self.last_maneuver_reason = 'target adjacent lane does not exist'
            return

        front_debug = self._front_debug_data()
        self.original_lane_id = int(current_wp.lane_id)
        self.original_road_id = int(current_wp.road_id)
        self.target_lane_id = int(target_wp.lane_id)
        self.tracked_front_actor_id = int(front_debug.get('actor_id', -1))
        self.overtake_start_location = self._ego_pos.location
        self.passed_front_vehicle = False
        self.return_lane_clear = False
        self.lane_change_steps = 0
        self.passing_steps = 0
        self.last_overtake_abort_reason = ''
        self.abort_route_injected = False
        self.overtake_state = self.CHANGING_LANE

        injected = self._inject_route_to_lane(
            target_wp,
            self.overtake_lookahead,
            decision.maneuver,
            'overtake lane-change route injected')
        if not injected:
            self.overtake_state = self.KEEP_LANE
            self._reset_overtake_context()
        else:
            self.overtake_attempt_count += 1

    def _run_overtake_state_machine(self):
        """Update overtaking state and inject return route when appropriate."""
        self.last_maneuver_applied = 'none'
        self.last_maneuver_reason = ''
        self.return_lane_clear = False

        if not self.overtake_enabled:
            self.overtake_state = self.KEEP_LANE
            return

        current_wp = self._map.get_waypoint(self._ego_pos.location)

        if self.overtake_state == self.OVERTAKE_DONE:
            if self.max_overtakes > 0 and \
                    self.completed_overtake_count >= self.max_overtakes:
                self.last_maneuver_reason = (
                    'configured maximum number of overtakes reached')
                return
            if self.overtake_cooldown > 0:
                self.overtake_cooldown -= 1
                self.last_maneuver_reason = (
                    'overtake cooldown before searching for next vehicle')
                return

            self._reset_overtake_context()
            self.overtake_state = self.KEEP_LANE
            self.last_maneuver_reason = 'ready for next overtaking cycle'
            return

        if self.overtake_state == self.KEEP_LANE:
            self._start_overtake_if_requested()
            return

        if self.overtake_state == self.CHANGING_LANE:
            self.lane_change_steps += 1
            if self.last_llm_decision is not None and \
                    self.last_llm_decision.maneuver == 'abort_overtake':
                self._begin_abort('LLM requested abort during lane change')
                return
            if self.max_lane_change_steps > 0 and \
                    self.lane_change_steps > self.max_lane_change_steps:
                self._begin_abort('lane change exceeded configured step limit')
                return
            if current_wp.lane_id == self.target_lane_id:
                self.overtake_state = self.PASSING
                self.passing_steps = 0
                self.last_maneuver_reason = 'lane change completed; passing front vehicle'
            else:
                self.last_maneuver_reason = 'changing lane toward overtake lane'
            return

        if self.overtake_state == self.PASSING:
            self.passing_steps += 1
            self.passed_front_vehicle = self._has_passed_tracked_front_vehicle()
            if not self.passed_front_vehicle:
                if self.last_llm_decision is not None and \
                        self.last_llm_decision.maneuver == 'abort_overtake':
                    self._begin_abort('LLM requested abort while passing')
                    return
                if current_wp.lane_id != self.target_lane_id:
                    self._begin_abort(
                        'ego left target lane before passing tracked vehicle')
                    return
                lane = self._lane_check_data()
                remaining = float(lane.get(
                    'current_lane_continuous_distance', 999.0))
                if remaining < self.min_remaining_lane_continuity:
                    self._begin_abort(
                        'target lane ends or merges in %.1fm while passing' %
                        remaining)
                    return
                if self.max_passing_steps > 0 and \
                        self.passing_steps > self.max_passing_steps:
                    self._begin_abort('passing exceeded configured step limit')
                    return
                if self.max_passing_distance > 0.0 and \
                        self._distance_from_overtake_start() > \
                        self.max_passing_distance:
                    self._begin_abort(
                        'passing exceeded configured distance limit')
                    return
                self.last_maneuver_reason = 'passing; front vehicle not safely behind yet'
                return

            self.return_lane_clear = self._lane_clear_for_original_return(current_wp)
            if not self.return_lane_clear:
                self.last_maneuver_reason = 'passed front vehicle, but original lane is not clear'
                return

            original_wp = self._find_original_lane_waypoint(current_wp)
            if original_wp is None:
                self.last_maneuver_reason = 'original lane waypoint not found for return'
                return

            self.overtake_state = self.RETURNING
            self._inject_route_to_lane(
                original_wp,
                self.return_lookahead,
                'return_to_original_lane',
                'return-to-original-lane route injected')
            return

        if self.overtake_state == self.ABORTING:
            if self._is_back_on_original_lane():
                self.overtake_state = self.OVERTAKE_DONE
                self.overtake_cooldown = max(
                    0, self.overtake_cooldown_steps)
                self.last_maneuver_applied = 'overtake_aborted'
                self.last_maneuver_reason = (
                    'aborted overtaking attempt returned to original lane')
                self._restore_original_destination()
                return

            if not self.abort_route_injected and \
                    self._lane_clear_for_original_return(current_wp):
                original_wp = self._find_original_lane_waypoint(current_wp)
                if original_wp is not None:
                    self.abort_route_injected = self._inject_route_to_lane(
                        original_wp,
                        self.return_lookahead,
                        'abort_overtake',
                        self.last_overtake_abort_reason)
            self.last_maneuver_reason = (
                self.last_overtake_abort_reason or
                'aborting overtaking attempt')
            return

        if self.overtake_state == self.RETURNING:
            if self._is_back_on_original_lane():
                self.overtake_state = self.OVERTAKE_DONE
                self.completed_overtake_count += 1
                self.overtake_cooldown = max(0, self.overtake_cooldown_steps)
                self.last_maneuver_applied = 'overtake_done'
                self.last_maneuver_reason = (
                    'ego returned to original lane; overtaking cycle completed '
                    'and original destination restored')
                self._restore_original_destination()
            else:
                self.last_maneuver_reason = 'returning to original lane'

    def run_step(self, target_speed=None, collision_detector_enabled=True,
                 lane_change_allowed=True):
        self._run_overtake_state_machine()

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
