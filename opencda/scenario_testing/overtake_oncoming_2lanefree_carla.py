# -*- coding: utf-8 -*-

import os
import copy
import carla
from omegaconf import OmegaConf

import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.core.plan.oncoming_overtake_agent import (
    CooperativeOvertakeAgent,
    WrongWayAgent,
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


def run_scenario(opt, scenario_params):
    scenario_manager = None
    eval_manager = None
    single_cav_list = []
    bg_veh_list = []

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
                "overtake_oncoming_2lanefree_carla.log", True
            )

        # create BOTH ego and oncoming through default OpenCDA pipeline
        # IMPORTANT: oncoming_cav yaw in YAML must be 0 at creation time
        single_cav_list = scenario_manager.create_vehicle_manager(
            application=['single'],
            data_dump=False
        )

        # create background traffic (lead slow NPC etc.)
        traffic_manager, bg_veh_list = scenario_manager.create_traffic_carla()

        if len(single_cav_list) < 2:
            raise RuntimeError(
                "This scenario requires exactly two CAVs in single_cav_list: "
                "ego_cav and oncoming_cav."
            )

        ego_vm = single_cav_list[0]
        oncoming_vm = single_cav_list[1]

        # replace default agents with custom agents
        ego_behavior_cfg = _merge_behavior_cfg(scenario_params, 0)
        ego_extra_cfg = _to_dict(
            scenario_params['scenario']['single_cav_list'][0].get('cooperative_overtake', {})
        )
        ego_vm.agent = CooperativeOvertakeAgent(
            ego_vm.vehicle,
            scenario_manager.carla_map,
            ego_behavior_cfg,
            ego_vm.v2x_manager,
            scenario_manager.cav_world,
            ego_extra_cfg
        )

        oncoming_behavior_cfg = _merge_behavior_cfg(scenario_params, 1)
        oncoming_extra_cfg = _to_dict(
            scenario_params['scenario']['single_cav_list'][1].get('wrong_way', {})
        )
        oncoming_vm.agent = WrongWayAgent(
            oncoming_vm.vehicle,
            scenario_manager.carla_map,
            oncoming_behavior_cfg,
            oncoming_extra_cfg
        )

        # reset ego destination for the new ego agent
        ego_dest_cfg = scenario_params['scenario']['single_cav_list'][0]['destination']
        ego_goal = _loc_from_xyz(ego_dest_cfg)

        ego_vm.update_info()
        ego_vm.set_destination(
            ego_vm.vehicle.get_location(),
            ego_goal,
            clean=True
        )

        # pass robust goal to custom ego agent
        ego_vm.agent.set_goal_location(ego_goal)

        # turn oncoming_cav into reverse-direction traffic AFTER creation succeeds
        oncoming_cfg = scenario_params['scenario']['single_cav_list'][1]
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

        print("single cav count =", len(single_cav_list))
        for vm in single_cav_list:
            print("CAV actor:", vm.vehicle.id, vm.vehicle.get_transform())

        eval_manager = EvaluationManager(
            scenario_manager.cav_world,
            script_name='overtake_oncoming_2lanefree_carla',
            current_time=scenario_params['current_time']
        )

        spectator = scenario_manager.world.get_spectator()
        max_steps = int(scenario_params['scenario'].get('max_steps', 5000))
        step = 0

        while step < max_steps:
            step += 1

            scenario_manager.tick()
            scenario_manager.cav_world.tick()

            if step == 1:
                print("AFTER FIRST TICK oncoming:", oncoming_vm.vehicle.get_transform())

            # spectator looks at the interaction center between ego and oncoming
            ego_tf = ego_vm.vehicle.get_transform()
            oncoming_tf = oncoming_vm.vehicle.get_transform()

            center_x = (ego_tf.location.x + oncoming_tf.location.x) / 2.0
            center_y = (ego_tf.location.y + oncoming_tf.location.y) / 2.0

            spectator.set_transform(
                carla.Transform(
                    carla.Location(x=center_x, y=center_y, z=120),
                    carla.Rotation(pitch=-90)
                )
            )

            # update all CAVs
            for vm in single_cav_list:
                vm.update_info()

            # run all CAVs
            for vm in single_cav_list:
                control = vm.run_step()
                vm.vehicle.apply_control(control)

            # ONLY ONE stop condition remains here
            if step > 500 and ego_vm.vehicle.get_location().distance(ego_goal) < 10.0:
                print("ego vehicle reaches the goal region, stop scenario.")
                break

    finally:
        if eval_manager is not None:
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

        for v in bg_veh_list:
            try:
                v.destroy()
            except Exception:
                pass