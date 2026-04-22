# -*- coding: utf-8 -*-
"""
Custom agents for cooperative overtaking with an oncoming vehicle
on 2lane_freeway_simplified.
"""

import carla

from opencda.core.plan.behavior_agent import BehaviorAgent
from opencda.core.common.misc import get_speed, cal_distance_angle


def normalize_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


class CooperativeOvertakeAgent(BehaviorAgent):
    """
    Ego agent for overtaking with an oncoming-traffic safety check.
    """

    def __init__(self,
                 vehicle,
                 carla_map,
                 behavior_cfg,
                 v2x_manager,
                 cav_world,
                 coop_cfg=None):
        super(CooperativeOvertakeAgent, self).__init__(vehicle, carla_map, behavior_cfg)

        coop_cfg = coop_cfg or {}

        self.v2x_manager = v2x_manager
        self.cav_world = cav_world

        self.use_v2v = coop_cfg.get('use_v2v', True)

        self.local_oncoming_detection_range = coop_cfg.get(
            'local_oncoming_detection_range', 45.0)
        self.v2v_oncoming_detection_range = coop_cfg.get(
            'v2v_oncoming_detection_range', 180.0)

        self.oncoming_yaw_threshold = coop_cfg.get(
            'oncoming_yaw_threshold', 120.0)

        self.lane_change_time = coop_cfg.get('lane_change_time', 1.4)
        self.extra_pass_distance = coop_cfg.get('extra_pass_distance', 18.0)
        self.oncoming_time_margin = coop_cfg.get('oncoming_time_margin', 2.8)
        self.min_relative_speed = coop_cfg.get('min_relative_speed', 2.0)
        self.return_distance_buffer = coop_cfg.get('return_distance_buffer', 12.0)

        self.min_run_steps = coop_cfg.get('min_run_steps', 500)
        self.goal_tolerance = coop_cfg.get('goal_tolerance', 10.0)
        self.goal_location = None
        self.internal_step = 0

        self.base_lane_id = None
        self.overtake_stage = 'IDLE'   # IDLE / OUT / RETURN
        self.lead_vehicle_id = None
        self.maneuver_target_speed = None

    def set_goal_location(self, goal_location):
        self.goal_location = goal_location

    def _close_to_goal(self):
        if self.goal_location is None:
            return False
        if self._ego_pos is None:
            return False
        if self.internal_step < self.min_run_steps:
            return False
        return self._ego_pos.location.distance(self.goal_location) < self.goal_tolerance

    def update_information(self, ego_pos, ego_speed, objects):
        super(CooperativeOvertakeAgent, self).update_information(
            ego_pos, ego_speed, objects)

        if self.base_lane_id is None and self._ego_pos is not None:
            self.base_lane_id = self._map.get_waypoint(
                self._ego_pos.location).lane_id

    def _is_oncoming_by_yaw(self, other_yaw):
        yaw_diff = abs(normalize_angle_deg(other_yaw - self._ego_pos.rotation.yaw))
        return yaw_diff >= self.oncoming_yaw_threshold

    def _nearest_oncoming_from_local(self):
        ego_wp = self._map.get_waypoint(self._ego_pos.location)
        best = None
        best_dist = 1e9

        for vehicle in self.obstacle_vehicles:
            loc = vehicle.get_location()
            dist, angle = cal_distance_angle(
                loc, self._ego_pos.location, self._ego_pos.rotation.yaw)

            if dist > self.local_oncoming_detection_range:
                continue
            if angle > 90.0:
                continue

            v_wp = self._map.get_waypoint(loc)
            if v_wp.lane_id == ego_wp.lane_id:
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
                    'location': loc
                }

        return best

    def _nearest_oncoming_from_v2v(self):
        if not self.use_v2v:
            return None

        ego_wp = self._map.get_waypoint(self._ego_pos.location)
        best = None
        best_dist = 1e9

        for vm in self.v2x_manager.cav_nearby:
            pos = vm.v2x_manager.get_ego_pos()
            speed = vm.v2x_manager.get_ego_speed()

            if pos is None or speed is None:
                continue

            dist, angle = cal_distance_angle(
                pos.location, self._ego_pos.location, self._ego_pos.rotation.yaw)

            if dist > self.v2v_oncoming_detection_range:
                continue
            if angle > 90.0:
                continue

            v_wp = self._map.get_waypoint(pos.location)
            if v_wp.lane_id == ego_wp.lane_id:
                continue

            if not self._is_oncoming_by_yaw(pos.rotation.yaw):
                continue

            if dist < best_dist:
                best_dist = dist
                best = {
                    'distance': dist,
                    'speed': speed,
                    'source': 'v2v',
                    'location': pos.location
                }

        return best

    def _nearest_oncoming_vehicle(self):
        cands = []

        local_cand = self._nearest_oncoming_from_local()
        if local_cand is not None:
            cands.append(local_cand)

        v2v_cand = self._nearest_oncoming_from_v2v()
        if v2v_cand is not None:
            cands.append(v2v_cand)

        if not cands:
            return None

        cands.sort(key=lambda x: x['distance'])
        return cands[0]

    def _estimate_overtake_window(self, lead_vehicle, lead_distance):
        lead_speed = get_speed(lead_vehicle)
        desired_speed = min(self.tailgate_speed, self.max_speed + 10.0)

        rel_speed = max(
            (desired_speed - lead_speed) / 3.6,
            self.min_relative_speed
        )

        d_clear = max(lead_distance, 0.0) + self.extra_pass_distance
        t_pass = 2.0 * self.lane_change_time + d_clear / rel_speed

        oncoming = self._nearest_oncoming_vehicle()
        if oncoming is None:
            return True, desired_speed, {
                't_pass': t_pass,
                't_oncoming': 9999.0,
                'source': 'none'
            }

        closing_speed = max((desired_speed + oncoming['speed']) / 3.6, 0.1)
        t_oncoming = oncoming['distance'] / closing_speed
        safe = t_oncoming > (t_pass + self.oncoming_time_margin)

        return safe, desired_speed, {
            't_pass': t_pass,
            't_oncoming': t_oncoming,
            'source': oncoming['source'],
            'oncoming_distance': oncoming['distance'],
            'oncoming_speed': oncoming['speed']
        }

    def _start_overtake(self, obstacle_vehicle):
        ego_wp = self._map.get_waypoint(self._ego_pos.location)
        left_wpt = ego_wp.get_left_lane()

        if left_wpt is None:
            return False
        if left_wpt.lane_type != carla.LaneType.Driving:
            return False
        if ego_wp.lane_id * left_wpt.lane_id <= 0:
            return False

        next_wpts = left_wpt.next(max(self._ego_speed / 3.6 * 8.0, 25.0))
        if len(next_wpts) == 0:
            return False

        target = next_wpts[0]
        self.overtake_stage = 'OUT'
        self.lead_vehicle_id = obstacle_vehicle.id
        self.overtake_counter = 100

        self.set_destination(
            self._ego_pos.location,
            target.transform.location,
            clean=True,
            end_reset=False
        )
        return True

    def _should_return_to_lane(self):
        if self.lead_vehicle_id is None:
            return False

        actor = self.vehicle.get_world().get_actors().find(self.lead_vehicle_id)
        if actor is None:
            return True

        dist, angle = cal_distance_angle(
            actor.get_location(),
            self._ego_pos.location,
            self._ego_pos.rotation.yaw
        )

        return angle > 100.0 and dist > self.return_distance_buffer

    def _start_return(self):
        cur_wp = self._map.get_waypoint(self._ego_pos.location)
        right_wpt = cur_wp.get_right_lane()

        if right_wpt is None:
            return False
        if right_wpt.lane_type != carla.LaneType.Driving:
            return False
        if cur_wp.lane_id * right_wpt.lane_id <= 0:
            return False

        next_wpts = right_wpt.next(max(self._ego_speed / 3.6 * 6.0, 20.0))
        if len(next_wpts) == 0:
            return False

        target = next_wpts[0]
        self.overtake_stage = 'RETURN'

        self.set_destination(
            self._ego_pos.location,
            target.transform.location,
            clean=True,
            end_reset=False
        )
        return True

    def _finish_return_if_needed(self):
        cur_wp = self._map.get_waypoint(self._ego_pos.location)

        if self.overtake_stage == 'RETURN' and cur_wp.lane_id == self.base_lane_id:
            self.overtake_stage = 'IDLE'
            self.lead_vehicle_id = None
            self.maneuver_target_speed = None

            self.set_destination(
                self._ego_pos.location,
                self.end_waypoint.transform.location,
                clean=True,
                end_reset=True,
                clean_history=True
            )

    def run_step(self,
                 target_speed=None,
                 collision_detector_enabled=True,
                 lane_change_allowed=True):
        self.internal_step += 1

        ego_vehicle_loc = self._ego_pos.location
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        self.ttc = 1000

        if self.overtake_counter > 0:
            self.overtake_counter -= 1

        if self.destination_push_flag > 0:
            self.destination_push_flag -= 1

        # only custom stop logic remains
        if self._close_to_goal():
            return 0, None

        if self.traffic_light_manager(ego_vehicle_wp) != 0:
            return 0, None

        self._finish_return_if_needed()

        if self.overtake_stage == 'IDLE' and \
                len(self.get_local_planner().get_waypoints_queue()) == 0 and \
                len(self.get_local_planner().get_waypoint_buffer()) <= 2:
            self.overtake_allowed = True and self.overtake_allowed_origin
            self.lane_change_allowed = True
            self.destination_push_flag = 0
            self.set_destination(
                ego_vehicle_loc,
                self.end_waypoint.transform.location,
                clean=True,
                clean_history=True
            )

        rx, ry, rk, ryaw = self._local_planner.generate_path()

        self.lane_change_allowed = self.check_lane_change_permission(
            lane_change_allowed, collision_detector_enabled, rk
        )

        is_hazard = False
        obstacle_vehicle = None
        distance = 1e9

        if collision_detector_enabled:
            is_hazard, obstacle_vehicle, distance = self.collision_manager(
                rx, ry, ryaw, ego_vehicle_wp
            )

        if self.overtake_stage == 'OUT':
            if self._should_return_to_lane():
                self._start_return()

            if is_hazard and distance < max(self.break_distance, 3.0):
                return 0, None

            chosen_speed = self.maneuver_target_speed or self.tailgate_speed
            chosen_speed, target_loc = self._local_planner.run_step(
                rx, ry, rk, target_speed=chosen_speed
            )
            return chosen_speed, target_loc

        if self.overtake_stage == 'RETURN':
            if is_hazard and distance < max(self.break_distance, 3.0):
                return 0, None

            chosen_speed = max(self.max_speed - self.speed_lim_dist, 35.0)
            chosen_speed, target_loc = self._local_planner.run_step(
                rx, ry, rk, target_speed=chosen_speed
            )
            return chosen_speed, target_loc

        car_following_flag = False

        if is_hazard and obstacle_vehicle is not None:
            safe_window, desired_speed, info = self._estimate_overtake_window(
                obstacle_vehicle, distance
            )

            if self.debug:
                print(
                    "[ego] safe_window=%s source=%s t_pass=%.2f t_oncoming=%.2f"
                    % (safe_window, info['source'], info['t_pass'], info['t_oncoming'])
                )

            if safe_window and self.overtake_allowed and self.lane_change_allowed:
                started = self._start_overtake(obstacle_vehicle)
                if started:
                    self.maneuver_target_speed = desired_speed
                    rx, ry, rk, ryaw = self._local_planner.generate_path()
                    chosen_speed, target_loc = self._local_planner.run_step(
                        rx, ry, rk, target_speed=desired_speed
                    )
                    return chosen_speed, target_loc

            car_following_flag = True

        if car_following_flag:
            if distance < max(self.break_distance, 3.0):
                return 0, None

            target_speed = self.car_following_manager(
                obstacle_vehicle, distance, target_speed
            )
            target_speed, target_loc = self._local_planner.run_step(
                rx, ry, rk, target_speed=target_speed
            )
            return target_speed, target_loc

        target_speed, target_loc = self._local_planner.run_step(
            rx, ry, rk,
            target_speed=self.max_speed - self.speed_lim_dist
            if target_speed is None else target_speed
        )
        return target_speed, target_loc


class WrongWayAgent(BehaviorAgent):
    """
    Keep the oncoming CAV on the left lane while driving in reverse direction.
    """

    def __init__(self, vehicle, carla_map, behavior_cfg, wrong_way_cfg=None):
        super(WrongWayAgent, self).__init__(vehicle, carla_map, behavior_cfg)
        wrong_way_cfg = wrong_way_cfg or {}
        self.target_speed_const = wrong_way_cfg.get('target_speed', 60.0)
        self.look_ahead = wrong_way_cfg.get('look_ahead', 8.0)

    def run_step(self,
                 target_speed=None,
                 collision_detector_enabled=True,
                 lane_change_allowed=True):
        if self._ego_pos is None:
            return 0, None

        cur_wp = self._map.get_waypoint(self._ego_pos.location)
        prev_wpts = cur_wp.previous(self.look_ahead)

        if len(prev_wpts) == 0:
            return 0, None

        target_wp = prev_wpts[0]
        chosen_speed = self.target_speed_const if target_speed is None else target_speed
        return chosen_speed, target_wp.transform.location