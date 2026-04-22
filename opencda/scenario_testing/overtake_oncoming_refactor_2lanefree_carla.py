# -*- coding: utf-8 -*-

import os
import copy
import carla
from omegaconf import OmegaConf

import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.core.plan.overtake_experiment_agent import (
    OvertakeExperimentAgent,
    SameLaneConstantAgent,
    WrongWayConstantAgent,
)
from opencda.scenario_testing.evaluations.evaluate_manager import EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time


def _to_dict(cfg):
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return copy.deepcopy(cfg)


def _merge_behavior_cfg(scenario_params, cav_index):
    base_behavior = _to_dict(scenario_params['vehicle_base']['behavior'])
    cav_cfg = scenario_params['scenario']['single_cav_list'][cav_index]
    cav_behavior = _to_dict(cav_cfg.get('behavior', {}))
    merged = OmegaConf.merge(base_behavior, cav_behavior)
    return _to_dict(merged)


def _loc_from_xyz(xyz):
    return carla.Location(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


def _latest_safety_flags(vm):
    q = vm.safety_manager.status_queue
    if q is None or len(q) == 0:
        return {}
    return q[-1][1]


def _left_drivable_map(carla_map, vehicle):
    wp = carla_map.get_waypoint(
        vehicle.get_location(),
        project_to_road=False,
        lane_type=carla.LaneType.Driving
    )
    return wp is None


def run_scenario(opt, scenario_params):
    scenario_manager = None
    eval_manager = None
    single_cav_list = []
    scenario_ok = False

    try:
        scenario_params = add_current_time(scenario_params)

        current_path = os.path.dirname(os.path.realpath(__file__))
        xodr_path = os.path.join(
            current_path,
            '../assets/2lane_freeway_simplified/2lane_freeway_simplified.xodr'
        )

        cav_world = CavWorld(opt.apply_ml)

        scenario_manager = sim_api.ScenarioManager(
            scenario_params,
            opt.apply_ml,
            opt.version,
            xodr_path=xodr_path,
            cav_world=cav_world
        )

        if opt.record:
            scenario_manager.client.start_recorder(
                "overtake_oncoming_refactor_2lanefree_carla.log", True
            )

        # 3 CAVs: ego, lead, oncoming
        single_cav_list = scenario_manager.create_vehicle_manager(
            application=['single'],
            data_dump=False
        )

        if len(single_cav_list) < 3:
            raise RuntimeError(
                "This scenario requires exactly three CAVs in single_cav_list: "
                "ego_cav, lead_cav, oncoming_cav."
            )

        ego_vm = single_cav_list[0]
        lead_vm = single_cav_list[1]
        oncoming_vm = single_cav_list[2]

        ego_behavior_cfg = _merge_behavior_cfg(scenario_params, 0)
        ego_exp_cfg = _to_dict(
            scenario_params['scenario']['single_cav_list'][0].get('experiment_agent', {})
        )
        ego_vm.agent = OvertakeExperimentAgent(
            ego_vm.vehicle,
            scenario_manager.carla_map,
            ego_behavior_cfg,
            ego_vm.v2x_manager,
            ego_exp_cfg
        )

        lead_cfg = _to_dict(scenario_params['scenario']['single_cav_list'][1])
        lead_agent_cfg = _to_dict(lead_cfg.get('lead_agent', {}))
        lead_vm.agent = SameLaneConstantAgent(
            lead_vm.vehicle,
            scenario_manager.carla_map,
            lead_agent_cfg
        )

        oncoming_cfg = _to_dict(scenario_params['scenario']['single_cav_list'][2])
        oncoming_agent_cfg = _to_dict(oncoming_cfg.get('wrong_way', {}))
        oncoming_vm.agent = WrongWayConstantAgent(
            oncoming_vm.vehicle,
            scenario_manager.carla_map,
            oncoming_agent_cfg
        )

        # rotate oncoming AFTER creation
        sp = oncoming_cfg['spawn_position']
        reverse_tf = carla.Transform(
            carla.Location(x=float(sp[0]), y=float(sp[1]), z=float(sp[2])),
            carla.Rotation(
                roll=float(sp[3]),
                yaw=180.0,
                pitch=float(sp[5])
            )
        )
        oncoming_vm.vehicle.set_transform(reverse_tf)

        ego_goal_cfg = scenario_params['scenario']['single_cav_list'][0]['destination']
        ego_goal = _loc_from_xyz(ego_goal_cfg)

        ego_vm.update_info()
        ego_vm.agent.set_goal_location(ego_goal)

        print("single cav count =", len(single_cav_list))
        for vm in single_cav_list:
            print("CAV actor:", vm.vehicle.id, vm.vehicle.get_transform())

        eval_manager = EvaluationManager(
            scenario_manager.cav_world,
            script_name='overtake_oncoming_refactor_2lanefree_carla',
            current_time=scenario_params['current_time']
        )

        spectator = scenario_manager.world.get_spectator()
        max_steps = int(scenario_params['scenario'].get('max_steps', 5000))
        min_run_steps = int(
            scenario_params['scenario']
            .get('single_cav_list')[0]
            .get('experiment_agent', {})
            .get('min_run_steps', 200)
        )
        step = 0

        while step < max_steps:
            step += 1

            scenario_manager.tick()
            scenario_manager.cav_world.tick()

            if step == 1:
                print("AFTER FIRST TICK oncoming:", oncoming_vm.vehicle.get_transform())

            # =========================================================
            # FIXED CAMERA:
            # hard lock top-down on ego CAV body
            # =========================================================
            ego_tf = ego_vm.vehicle.get_transform()
            spectator.set_transform(
                carla.Transform(
                    carla.Location(
                        x=ego_tf.location.x,
                        y=ego_tf.location.y,
                        z=25.0
                    ),
                    carla.Rotation(
                        pitch=-90.0,
                        yaw=0.0,
                        roll=0.0
                    )
                )
            )

            for vm in single_cav_list:
                vm.update_info()

            for vm in single_cav_list:
                control = vm.run_step()
                vm.vehicle.apply_control(control)

            if step % 20 == 0:
                print(
                    "[step %d] ego=(%.2f, %.2f) lead=(%.2f, %.2f) oncoming=(%.2f, %.2f) ego_state=%s"
                    % (
                        step,
                        ego_vm.vehicle.get_location().x,
                        ego_vm.vehicle.get_location().y,
                        lead_vm.vehicle.get_location().x,
                        lead_vm.vehicle.get_location().y,
                        oncoming_vm.vehicle.get_location().x,
                        oncoming_vm.vehicle.get_location().y,
                        ego_vm.agent.state
                    )
                )

            terminate = False
            terminate_reason = None

            for vm in single_cav_list:
                flags = _latest_safety_flags(vm)

                if flags.get('collision', False):
                    terminate = True
                    terminate_reason = "collision"
                    break

                if flags.get('offroad', False):
                    terminate = True
                    terminate_reason = "offroad"
                    break

                if flags.get('stuck', False):
                    terminate = True
                    terminate_reason = "stuck"
                    break

            if not terminate:
                for vm in single_cav_list:
                    if _left_drivable_map(scenario_manager.carla_map, vm.vehicle):
                        terminate = True
                        terminate_reason = "left_drivable_map"
                        break

            if terminate:
                print("terminate scenario due to:", terminate_reason)
                break

            if step > min_run_steps and ego_vm.vehicle.get_location().distance(ego_goal) < 10.0:
                print("ego vehicle reaches the goal region, stop scenario.")
                break

        print("scenario finished at step =", step)
        scenario_ok = True

    finally:
        if scenario_ok and eval_manager is not None:
            eval_manager.evaluate()

        if opt.record and scenario_manager is not None:
            scenario_manager.client.stop_recorder()

        if scenario_manager is not None:
            scenario_manager.close()

        for v in single_cav_list:
            try:
                v.destroy()
            except Exception:
                pass