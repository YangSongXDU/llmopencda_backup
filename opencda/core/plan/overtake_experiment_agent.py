# -*- coding: utf-8 -*-

import carla
import matplotlib.pyplot as plt

from opencda.core.common.misc import get_speed, cal_distance_angle
from opencda.customize.llm_gateway import LLMSemanticGateway


def normalize_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


class DummyDebugHelper(object):
    def __init__(self, name):
        self.name = name

    def evaluate(self):
        fig, ax = plt.subplots()
        ax.set_title(self.name)
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.plot([], [])
        txt = "%s: custom experiment agent, no built-in debug curves." % self.name
        return fig, txt


class BaseExperimentAgent(object):
    def __init__(self, vehicle, carla_map, config):
        self.vehicle = vehicle
        self._map = carla_map
        self.config = config or {}

        self._ego_pos = None
        self._ego_speed = 0.0  # km/h
        self.objects = {'vehicles': [], 'traffic_lights': []}

        self.initial_global_route = []
        self.debug_helper = DummyDebugHelper(self.__class__.__name__)

    def update_information(self, ego_pos, ego_speed, objects):
        self._ego_pos = ego_pos
        self._ego_speed = ego_speed
        self.objects = objects if objects is not None else {
            'vehicles': [],
            'traffic_lights': []
        }

    def run_step(self, target_speed=None):
        raise NotImplementedError


class SameLaneConstantAgent(BaseExperimentAgent):
    def __init__(self, vehicle, carla_map, config):
        super(SameLaneConstantAgent, self).__init__(vehicle, carla_map, config)
        self.target_speed_const = config.get('target_speed', 8.0)  # km/h
        self.look_ahead = config.get('look_ahead', 8.0)

    def run_step(self, target_speed=None):
        if self._ego_pos is None:
            return 0.0, None

        cur_wp = self._map.get_waypoint(
            self._ego_pos.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        next_wps = cur_wp.next(self.look_ahead)
        if len(next_wps) == 0:
            return 0.0, None

        target_wp = next_wps[0]
        chosen_speed = self.target_speed_const if target_speed is None else target_speed
        return chosen_speed, target_wp.transform.location


class WrongWayConstantAgent(BaseExperimentAgent):
    def __init__(self, vehicle, carla_map, config):
        super(WrongWayConstantAgent, self).__init__(vehicle, carla_map, config)
        self.target_speed_const = config.get('target_speed', 15.0)  # km/h
        self.look_ahead = config.get('look_ahead', 8.0)

    def run_step(self, target_speed=None):
        if self._ego_pos is None:
            return 0.0, None

        cur_wp = self._map.get_waypoint(
            self._ego_pos.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        prev_wps = cur_wp.previous(self.look_ahead)
        if len(prev_wps) == 0:
            return 0.0, None

        target_wp = prev_wps[0]
        chosen_speed = self.target_speed_const if target_speed is None else target_speed
        return chosen_speed, target_wp.transform.location


class OvertakeExperimentAgent(BaseExperimentAgent):
    """
    ego agent with safer staged overtaking:

    FOLLOW
    EVALUATE
    CHANGE_LEFT
    PASS_LEFT
    RETURN_RIGHT
    ABORT_RIGHT
    CRUISE
    """

    def __init__(self, vehicle, carla_map, behavior_cfg, v2x_manager, exp_cfg):
        raw_cfg = exp_cfg or {}
        agent_cfg = raw_cfg.get('experiment_agent', raw_cfg)

        super(OvertakeExperimentAgent, self).__init__(vehicle, carla_map, agent_cfg)

        self.v2x_manager = v2x_manager

        self.max_speed = behavior_cfg.get('max_speed', 55.0)
        self.tailgate_speed = behavior_cfg.get('tailgate_speed', 65.0)
        self.debug = behavior_cfg.get('local_planner', {}).get('debug', False)

        self.use_v2v = agent_cfg.get('use_v2v', True)

        self.local_oncoming_detection_range = agent_cfg.get(
            'local_oncoming_detection_range', 80.0)
        self.v2v_oncoming_detection_range = agent_cfg.get(
            'v2v_oncoming_detection_range', 250.0)
        self.oncoming_yaw_threshold = agent_cfg.get(
            'oncoming_yaw_threshold', 120.0)

        self.follow_speed_margin = agent_cfg.get('follow_speed_margin', 2.0)
        self.follow_min_speed = agent_cfg.get('follow_min_speed', 6.0)
        self.cruise_speed = agent_cfg.get('cruise_speed', 35.0)

        self.look_ahead_current = agent_cfg.get('look_ahead_current', 10.0)
        self.look_ahead_left = agent_cfg.get('look_ahead_left', 18.0)
        self.look_ahead_return = agent_cfg.get('look_ahead_return', 14.0)

        self.overtake_trigger_distance = agent_cfg.get('overtake_trigger_distance', 30.0)
        self.emergency_stop_distance = agent_cfg.get('emergency_stop_distance', 4.0)
        self.hard_brake_distance = agent_cfg.get('hard_brake_distance', 8.0)
        self.soft_follow_distance = agent_cfg.get('soft_follow_distance', 14.0)

        self.right_lane_center_y = agent_cfg.get('right_lane_center_y', 4.8)
        self.left_lane_center_y = agent_cfg.get('left_lane_center_y', 8.3)
        self.lane_y_tol = agent_cfg.get('lane_y_tolerance', 1.5)

        self.change_left_entry_speed = agent_cfg.get('change_left_entry_speed', 18.0)
        self.pass_left_speed = agent_cfg.get('pass_left_speed', 45.0)
        self.return_right_speed = agent_cfg.get('return_right_speed', 30.0)
        self.abort_right_speed = agent_cfg.get('abort_right_speed', 18.0)

        self.return_headway_buffer = agent_cfg.get('return_headway_buffer', 18.0)
        self.left_lane_reached_tol = agent_cfg.get('left_lane_reached_tol', 0.5)

        self.lane_change_time = agent_cfg.get('lane_change_time', 1.8)
        self.extra_pass_distance = agent_cfg.get('extra_pass_distance', 18.0)
        self.oncoming_time_margin = agent_cfg.get('oncoming_time_margin', 2.5)
        self.min_relative_speed = agent_cfg.get('min_relative_speed', 2.0)
        self.left_settle_time = agent_cfg.get('left_settle_time', 1.0)
        self.right_settle_time = agent_cfg.get('right_settle_time', 1.0)
        self.pass_prepare_time = agent_cfg.get('pass_prepare_time', 0.8)

        self.return_oncoming_min_distance = agent_cfg.get(
            'return_oncoming_min_distance', 28.0)
        self.return_oncoming_time_margin = agent_cfg.get(
            'return_oncoming_time_margin', 2.5)

        self.abort_change_left_min_distance = agent_cfg.get(
            'abort_change_left_min_distance', 22.0)
        self.abort_change_left_time_margin = agent_cfg.get(
            'abort_change_left_time_margin', 2.0)
        self.abort_pass_left_time_margin = agent_cfg.get(
            'abort_pass_left_time_margin', 1.5)

        self.goal_location = None
        self.state = 'FOLLOW'
        self.lead_vehicle_id = None

        llm_cfg = agent_cfg.get('llm', {})

        self.llm_enabled = llm_cfg.get('enabled', False)
        self.llm_endpoint = llm_cfg.get(
            'endpoint', 'http://127.0.0.1:8712/reason/overtake')
        self.llm_timeout = llm_cfg.get('timeout', 5.0)
        self.llm_cooldown_steps = llm_cfg.get('cooldown_steps', 8)
        self.llm_min_confidence = llm_cfg.get('min_confidence', 0.55)

        self.llm_use_in_evaluate = llm_cfg.get('use_in_evaluate', True)
        self.llm_use_in_change_left = llm_cfg.get('use_in_change_left', True)
        self.llm_use_in_pass_left = llm_cfg.get('use_in_pass_left', True)
        self.llm_use_in_return_right = llm_cfg.get('use_in_return_right', False)

        self.llm_gateway = LLMSemanticGateway(
            endpoint=self.llm_endpoint,
            timeout=self.llm_timeout,
            enabled=self.llm_enabled
        )

        self._internal_step = 0
        self._last_llm_state = None
        self._last_llm_query_step = -999999
        self._last_llm_response = None

    def set_goal_location(self, goal_location):
        self.goal_location = goal_location
        if self._ego_pos is not None:
            self.initial_global_route = self._build_initial_route(
                self._ego_pos.location, goal_location
            )

    def _build_initial_route(self, start_location, goal_location, step=5.0, max_iter=400):
        route = []
        try:
            cur_wp = self._map.get_waypoint(
                start_location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
        except Exception:
            return route

        route.append((cur_wp, None))
        last_wp = cur_wp

        for _ in range(max_iter):
            if last_wp.transform.location.distance(goal_location) < step:
                break

            nxt = last_wp.next(step)
            if len(nxt) == 0:
                break

            last_wp = nxt[0]
            route.append((last_wp, None))

        return route

    def _target_wp_by_lane_center(self, target_y, forward_distance):
        probe_loc = carla.Location(
            x=self._ego_pos.location.x + forward_distance,
            y=target_y,
            z=self._ego_pos.location.z
        )
        return self._map.get_waypoint(
            probe_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

    def _current_on_right_lane(self):
        return abs(self._ego_pos.location.y - self.right_lane_center_y) < self.lane_y_tol

    def _current_on_left_lane(self):
        return abs(self._ego_pos.location.y - self.left_lane_center_y) < self.lane_y_tol

    def _left_lane_reached(self):
        return abs(self._ego_pos.location.y - self.left_lane_center_y) < self.left_lane_reached_tol

    def _is_oncoming_by_yaw(self, other_yaw):
        yaw_diff = abs(normalize_angle_deg(other_yaw - self._ego_pos.rotation.yaw))
        return yaw_diff >= self.oncoming_yaw_threshold

    def _find_lead_vehicle_same_lane(self):
        best_vehicle = None
        best_dist = 1e9
        ego_x = self._ego_pos.location.x

        for vehicle in self.objects.get('vehicles', []):
            loc = vehicle.get_location()

            if abs(loc.y - self.right_lane_center_y) > self.lane_y_tol:
                continue
            if loc.x <= ego_x:
                continue

            dist = loc.x - ego_x
            if dist < best_dist:
                best_dist = dist
                best_vehicle = vehicle

        return best_vehicle, best_dist

    def _find_oncoming_local(self):
        best = None
        best_dist = 1e9

        for vehicle in self.objects.get('vehicles', []):
            loc = vehicle.get_location()
            dist, angle = cal_distance_angle(
                loc, self._ego_pos.location, self._ego_pos.rotation.yaw
            )

            if abs(loc.y - self.left_lane_center_y) > self.lane_y_tol:
                continue
            if dist > self.local_oncoming_detection_range:
                continue
            if angle > 90.0:
                continue

            try:
                other_yaw = vehicle.get_transform().rotation.yaw
            except Exception:
                continue

            if not self._is_oncoming_by_yaw(other_yaw):
                continue

            if dist < best_dist:
                best_dist = dist
                best = {
                    'distance': dist,
                    'speed': get_speed(vehicle),
                    'source': 'local',
                    'location': loc,
                    'yaw_deg': other_yaw
                }

        return best

    def _find_oncoming_v2v(self):
        if not self.use_v2v:
            return None

        nearby = getattr(self.v2x_manager, "cav_nearby", {})
        for_vm = nearby.values() if hasattr(nearby, "values") else []

        best = None
        best_dist = 1e9

        for vm in for_vm:
            try:
                pos = vm.v2x_manager.get_ego_pos()
                spd = vm.v2x_manager.get_ego_speed()
            except Exception:
                continue

            if pos is None or spd is None:
                continue

            dist, angle = cal_distance_angle(
                pos.location, self._ego_pos.location, self._ego_pos.rotation.yaw
            )

            if abs(pos.location.y - self.left_lane_center_y) > self.lane_y_tol:
                continue
            if dist > self.v2v_oncoming_detection_range:
                continue
            if angle > 90.0:
                continue
            if not self._is_oncoming_by_yaw(pos.rotation.yaw):
                continue

            if dist < best_dist:
                best_dist = dist
                best = {
                    'distance': dist,
                    'speed': spd,
                    'source': 'v2v',
                    'location': pos.location,
                    'yaw_deg': pos.rotation.yaw
                }

        return best

    def _find_nearest_oncoming(self):
        cands = []

        local_cand = self._find_oncoming_local()
        if local_cand is not None:
            cands.append(local_cand)

        v2v_cand = self._find_oncoming_v2v()
        if v2v_cand is not None:
            cands.append(v2v_cand)

        if not cands:
            return None

        cands.sort(key=lambda x: x['distance'])
        return cands[0]

    def _estimate_oncoming_time(self, ego_planned_speed_kmh):
        oncoming = self._find_nearest_oncoming()
        if oncoming is None:
            return 9999.0, None

        closing_speed = max((ego_planned_speed_kmh + oncoming['speed']) / 3.6, 0.1)
        t_oncoming = oncoming['distance'] / closing_speed
        return t_oncoming, oncoming

    def _estimate_window(self, lead_vehicle, lead_distance):
        lead_speed = get_speed(lead_vehicle)
        desired_speed = min(self.pass_left_speed, self.tailgate_speed, self.max_speed)

        rel_speed = max((desired_speed - lead_speed) / 3.6, self.min_relative_speed)
        d_clear = max(lead_distance, 0.0) + self.extra_pass_distance

        longitudinal_pass_time = d_clear / rel_speed

        t_pass = (
            self.lane_change_time +
            self.left_settle_time +
            self.pass_prepare_time +
            longitudinal_pass_time +
            self.right_settle_time +
            self.lane_change_time
        )

        t_oncoming, oncoming = self._estimate_oncoming_time(desired_speed)

        if oncoming is None:
            return True, desired_speed, {
                't_pass': t_pass,
                't_oncoming': 9999.0,
                'source': 'none'
            }, None

        safe = t_oncoming > (t_pass + self.oncoming_time_margin)

        return safe, desired_speed, {
            't_pass': t_pass,
            't_oncoming': t_oncoming,
            'source': oncoming['source']
        }, oncoming

    def _lead_actor(self):
        if self.lead_vehicle_id is None:
            return None
        try:
            return self.vehicle.get_world().get_actors().find(self.lead_vehicle_id)
        except Exception:
            return None

    def _ego_clearly_ahead_of_lead(self):
        actor = self._lead_actor()
        if actor is None:
            return True
        return self._ego_pos.location.x > (
            actor.get_location().x + self.return_headway_buffer
        )

    def _safe_to_return_right(self):
        t_oncoming, oncoming = self._estimate_oncoming_time(self.return_right_speed)

        if oncoming is None:
            return True

        if oncoming['distance'] < self.return_oncoming_min_distance:
            return False

        if t_oncoming < self.return_oncoming_time_margin:
            return False

        return True

    def _need_abort_change_left(self):
        t_oncoming, oncoming = self._estimate_oncoming_time(self.change_left_entry_speed)

        if oncoming is None:
            return False

        if oncoming['distance'] < self.abort_change_left_min_distance:
            return True

        if t_oncoming < self.abort_change_left_time_margin:
            return True

        return False

    def _need_abort_pass_left(self):
        t_oncoming, oncoming = self._estimate_oncoming_time(self.pass_left_speed)

        if oncoming is None:
            return False

        if t_oncoming < self.abort_pass_left_time_margin and not self._ego_clearly_ahead_of_lead():
            return True

        return False

    def _should_query_llm(self, state_name):
        if not self.llm_enabled:
            return False

        state_allowed = False
        if state_name == 'EVALUATE' and self.llm_use_in_evaluate:
            state_allowed = True
        elif state_name == 'CHANGE_LEFT' and self.llm_use_in_change_left:
            state_allowed = True
        elif state_name == 'PASS_LEFT' and self.llm_use_in_pass_left:
            state_allowed = True
        elif state_name == 'RETURN_RIGHT' and self.llm_use_in_return_right:
            state_allowed = True

        if not state_allowed:
            return False

        if self._last_llm_state != state_name:
            return True

        if (self._internal_step - self._last_llm_query_step) >= self.llm_cooldown_steps:
            return True

        return False

    def _build_v2v_semantic_messages(self, oncoming_info, t_oncoming):
        msgs = []

        if oncoming_info is not None and oncoming_info.get('source') == 'v2v':
            msgs.append({
                "sender": "oncoming_cav",
                "lane_occupancy_next_8s": "occupied",
                "time_to_conflict_zone_s": round(float(t_oncoming), 2),
                "intent": "keep_lane"
            })

        return msgs

    def _build_scene_payload(self, ego_state, lead_vehicle, lead_distance,
                             t_pass, t_oncoming, rule_safe_now, oncoming_info):
        ego_lane_label = "left_opposite" if self._current_on_left_lane() else "right"

        lead_payload = None
        if lead_vehicle is not None:
            lead_loc = lead_vehicle.get_location()
            lead_tf = lead_vehicle.get_transform()
            lead_payload = {
                "present": True,
                "x": float(lead_loc.x),
                "y": float(lead_loc.y),
                "speed_kmh": float(get_speed(lead_vehicle)),
                "yaw_deg": float(lead_tf.rotation.yaw),
                "lane_label": "right"
            }

        oncoming_payload = None
        if oncoming_info is not None:
            oncoming_payload = {
                "present": True,
                "x": float(oncoming_info["location"].x),
                "y": float(oncoming_info["location"].y),
                "speed_kmh": float(oncoming_info["speed"]),
                "yaw_deg": float(oncoming_info["yaw_deg"]),
                "lane_label": "left_opposite"
            }

        payload = {
            "scenario_id": "overtake_oncoming_refactor_2lanefree_carla",
            "tick": int(self._internal_step),
            "ego": {
                "present": True,
                "x": float(self._ego_pos.location.x),
                "y": float(self._ego_pos.location.y),
                "speed_kmh": float(self._ego_speed),
                "yaw_deg": float(self._ego_pos.rotation.yaw),
                "lane_label": ego_lane_label
            },
            "lead": lead_payload,
            "oncoming": oncoming_payload,
            "local_semantics": {
                "ego_state": ego_state,
                "right_lane_center_y": float(self.right_lane_center_y),
                "left_lane_center_y": float(self.left_lane_center_y),
                "road_type": "two_lane_bidirectional_straight",
                "lane_change_required_for_overtake": True
            },
            "rule_metrics": {
                "lead_distance_m": None if lead_distance is None else float(lead_distance),
                "t_pass_s": None if t_pass is None else float(t_pass),
                "t_oncoming_s": None if t_oncoming is None else float(t_oncoming),
                "rule_safe_now": None if rule_safe_now is None else bool(rule_safe_now),
                "return_headway_buffer_m": float(self.return_headway_buffer)
            },
            "v2v_semantics": {
                "enabled": bool(self.use_v2v),
                "source": None if oncoming_info is None else oncoming_info.get("source"),
                "received_messages": self._build_v2v_semantic_messages(oncoming_info, t_oncoming if t_oncoming is not None else 9999.0)
            }
        }

        return payload

    def _query_llm_decision(self, ego_state, lead_vehicle, lead_distance,
                            t_pass, t_oncoming, rule_safe_now, oncoming_info):
        if not self.llm_enabled:
            if self.debug:
                print("[ego][LLM] disabled by config")
            return None

        if not self._should_query_llm(ego_state):
            if self.debug:
                print("[ego][LLM] skipped: cooldown/state policy, reuse last response")
            return self._last_llm_response

        payload = self._build_scene_payload(
            ego_state=ego_state,
            lead_vehicle=lead_vehicle,
            lead_distance=lead_distance,
            t_pass=t_pass,
            t_oncoming=t_oncoming,
            rule_safe_now=rule_safe_now,
            oncoming_info=oncoming_info
        )

        resp = self.llm_gateway.reason_overtake(payload)

        self._last_llm_state = ego_state
        self._last_llm_query_step = self._internal_step
        self._last_llm_response = resp

        if self.debug:
            if resp is None:
                print("[ego][LLM] request failed -> fallback to rule")
            else:
                print("[ego][LLM] state=%s decision=%s confidence=%s next=%s reason=%s"
                      % (
                          ego_state,
                          resp.get("decision"),
                          resp.get("confidence"),
                          resp.get("next_state_hint"),
                          resp.get("reason")
                      ))

        return resp

    def _llm_confident_enough(self, llm_resp):
        if llm_resp is None:
            return False
        try:
            conf = float(llm_resp.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        return conf >= self.llm_min_confidence

    def _apply_llm_evaluate(self, llm_resp, rule_safe):
        if llm_resp is None:
            return None

        if not self._llm_confident_enough(llm_resp):
            return None

        decision = llm_resp.get("decision", "")

        if decision == "ALLOW_OVERTAKE":
            if rule_safe:
                return "CHANGE_LEFT"
            return "FOLLOW"

        if decision == "DELAY_OVERTAKE":
            return "FOLLOW"

        if decision == "ABORT_AND_FOLLOW":
            return "FOLLOW"

        if decision == "RETURN_RIGHT_NOW":
            return "FOLLOW"

        if decision == "HOLD_LEFT_UNTIL_CLEAR":
            return "FOLLOW"

        return None

    def _apply_llm_change_left(self, llm_resp):
        if llm_resp is None:
            return None

        if not self._llm_confident_enough(llm_resp):
            return None

        decision = llm_resp.get("decision", "")
        next_state_hint = llm_resp.get("next_state_hint", "")

        if decision == "ABORT_AND_FOLLOW":
            return "ABORT_RIGHT"

        if decision == "RETURN_RIGHT_NOW":
            return "ABORT_RIGHT"

        if decision in ["DELAY_OVERTAKE", "ALLOW_OVERTAKE", "HOLD_LEFT_UNTIL_CLEAR"]:
            return "CHANGE_LEFT"

        if next_state_hint == "CHANGE_LEFT":
            return "CHANGE_LEFT"

        return None

    def _apply_llm_pass_left(self, llm_resp):
        if llm_resp is None:
            return None

        if not self._llm_confident_enough(llm_resp):
            return None

        decision = llm_resp.get("decision", "")

        if decision == "RETURN_RIGHT_NOW":
            return "RETURN_RIGHT"

        if decision == "ABORT_AND_FOLLOW":
            return "ABORT_RIGHT"

        if decision in ["HOLD_LEFT_UNTIL_CLEAR", "DELAY_OVERTAKE", "ALLOW_OVERTAKE"]:
            return "PASS_LEFT"

        return None

    def run_step(self, target_speed=None):
        self._internal_step += 1

        if self._ego_pos is None:
            return 0.0, None

        lead_vehicle, lead_distance = self._find_lead_vehicle_same_lane()

        if lead_vehicle is not None and self._current_on_right_lane():
            if lead_distance < self.emergency_stop_distance:
                target_wp = self._target_wp_by_lane_center(self.right_lane_center_y, 6.0)
                return 0.0, target_wp.transform.location

            if lead_distance < self.hard_brake_distance:
                target_wp = self._target_wp_by_lane_center(self.right_lane_center_y, 8.0)
                return 3.0, target_wp.transform.location

        if self.state in ['FOLLOW', 'CRUISE', 'EVALUATE']:
            if lead_vehicle is None:
                self.state = 'CRUISE'
            else:
                if lead_distance < self.overtake_trigger_distance:
                    self.state = 'EVALUATE'
                    safe, desired_speed, info, oncoming_info = self._estimate_window(
                        lead_vehicle, lead_distance)

                    if self.debug:
                        print("[ego] state=EVALUATE safe=%s source=%s t_pass=%.2f t_oncoming=%.2f lead_dist=%.2f"
                              % (
                                  safe,
                                  info['source'],
                                  info['t_pass'],
                                  info['t_oncoming'],
                                  lead_distance
                              ))

                    llm_resp = self._query_llm_decision(
                        ego_state='EVALUATE',
                        lead_vehicle=lead_vehicle,
                        lead_distance=lead_distance,
                        t_pass=info['t_pass'],
                        t_oncoming=info['t_oncoming'],
                        rule_safe_now=safe,
                        oncoming_info=oncoming_info
                    )

                    llm_action = self._apply_llm_evaluate(llm_resp, safe)

                    if llm_action == "CHANGE_LEFT":
                        self.state = 'CHANGE_LEFT'
                        self.lead_vehicle_id = getattr(
                            lead_vehicle, 'id',
                            getattr(lead_vehicle, 'carla_id', -1)
                        )
                    elif llm_action == "FOLLOW":
                        self.state = 'FOLLOW'
                    else:
                        if safe:
                            self.state = 'CHANGE_LEFT'
                            self.lead_vehicle_id = getattr(
                                lead_vehicle, 'id',
                                getattr(lead_vehicle, 'carla_id', -1)
                            )
                        else:
                            self.state = 'FOLLOW'
                else:
                    self.state = 'FOLLOW'

        elif self.state == 'CHANGE_LEFT':
            t_oncoming, oncoming_info = self._estimate_oncoming_time(self.change_left_entry_speed)
            llm_resp = self._query_llm_decision(
                ego_state='CHANGE_LEFT',
                lead_vehicle=lead_vehicle,
                lead_distance=lead_distance,
                t_pass=None,
                t_oncoming=t_oncoming,
                rule_safe_now=None,
                oncoming_info=oncoming_info
            )
            llm_action = self._apply_llm_change_left(llm_resp)

            if llm_action == "ABORT_RIGHT":
                self.state = 'ABORT_RIGHT'
            elif self._need_abort_change_left():
                self.state = 'ABORT_RIGHT'
            elif self._left_lane_reached():
                self.state = 'PASS_LEFT'
            else:
                self.state = 'CHANGE_LEFT'

        elif self.state == 'PASS_LEFT':
            t_oncoming, oncoming_info = self._estimate_oncoming_time(self.pass_left_speed)
            llm_resp = self._query_llm_decision(
                ego_state='PASS_LEFT',
                lead_vehicle=lead_vehicle,
                lead_distance=lead_distance,
                t_pass=None,
                t_oncoming=t_oncoming,
                rule_safe_now=None,
                oncoming_info=oncoming_info
            )
            llm_action = self._apply_llm_pass_left(llm_resp)

            if llm_action == "RETURN_RIGHT":
                if self._safe_to_return_right():
                    self.state = 'RETURN_RIGHT'
                else:
                    self.state = 'PASS_LEFT'
            elif llm_action == "ABORT_RIGHT":
                self.state = 'ABORT_RIGHT'
            elif self._need_abort_pass_left():
                if self._ego_clearly_ahead_of_lead() and self._safe_to_return_right():
                    self.state = 'RETURN_RIGHT'
                else:
                    self.state = 'ABORT_RIGHT'
            elif self._ego_clearly_ahead_of_lead():
                if self._safe_to_return_right():
                    self.state = 'RETURN_RIGHT'
                else:
                    self.state = 'PASS_LEFT'
            else:
                self.state = 'PASS_LEFT'

        elif self.state == 'RETURN_RIGHT':
            if self._current_on_right_lane():
                if lead_vehicle is None:
                    self.state = 'CRUISE'
                else:
                    self.state = 'FOLLOW'
                self.lead_vehicle_id = None

        elif self.state == 'ABORT_RIGHT':
            if self._current_on_right_lane():
                if lead_vehicle is None:
                    self.state = 'CRUISE'
                else:
                    self.state = 'FOLLOW'
                self.lead_vehicle_id = None

        if self.state == 'CRUISE':
            target_wp = self._target_wp_by_lane_center(
                self.right_lane_center_y, self.look_ahead_current
            )
            return self.cruise_speed, target_wp.transform.location

        if self.state == 'FOLLOW':
            if lead_vehicle is not None:
                if lead_distance < self.soft_follow_distance:
                    follow_speed = max(
                        self.follow_min_speed,
                        min(get_speed(lead_vehicle) - 1.0, self.cruise_speed - 5.0)
                    )
                else:
                    follow_speed = max(
                        self.follow_min_speed,
                        min(get_speed(lead_vehicle) + self.follow_speed_margin, self.cruise_speed)
                    )
            else:
                follow_speed = self.cruise_speed

            target_wp = self._target_wp_by_lane_center(
                self.right_lane_center_y, self.look_ahead_current
            )
            return follow_speed, target_wp.transform.location

        if self.state == 'CHANGE_LEFT':
            target_wp = self._target_wp_by_lane_center(
                self.left_lane_center_y, self.look_ahead_left
            )
            return self.change_left_entry_speed, target_wp.transform.location

        if self.state == 'PASS_LEFT':
            target_wp = self._target_wp_by_lane_center(
                self.left_lane_center_y, self.look_ahead_left
            )
            return self.pass_left_speed, target_wp.transform.location

        if self.state == 'RETURN_RIGHT':
            target_wp = self._target_wp_by_lane_center(
                self.right_lane_center_y, self.look_ahead_return
            )
            return self.return_right_speed, target_wp.transform.location

        if self.state == 'ABORT_RIGHT':
            target_wp = self._target_wp_by_lane_center(
                self.right_lane_center_y, self.look_ahead_return
            )
            return self.abort_right_speed, target_wp.transform.location

        target_wp = self._target_wp_by_lane_center(
            self.right_lane_center_y, self.look_ahead_current
        )
        return self.cruise_speed, target_wp.transform.location