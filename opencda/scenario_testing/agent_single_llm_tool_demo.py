# -*- coding: utf-8 -*-
"""
Scenario testing: single CAV demo for LLM Sensor-as-Tools Agent.
"""

import os

import carla

import opencda.scenario_testing.utils.sim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api

from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import \
    add_current_time
from opencda.customize.core.common.llm_experiment_recorder import \
    LLMExperimentRecorder


def run_scenario(opt, scenario_params):
    scenario_manager = None
    eval_manager = None
    single_cav_list = []
    bg_veh_list = []
    recorder = None

    try:
        scenario_params = add_current_time(scenario_params)
        current_path = os.path.dirname(os.path.realpath(__file__))
        xodr_path = os.path.join(
            current_path,
            '../assets/2lane_freeway_simplified/2lane_freeway_simplified.xodr')

        cav_world = CavWorld(opt.apply_ml)
        scenario_manager = sim_api.ScenarioManager(
            scenario_params,
            opt.apply_ml,
            opt.version,
            xodr_path=xodr_path,
            cav_world=cav_world)

        if opt.record:
            scenario_manager.client.start_recorder(
                'agent_single_llm_tool_demo.log', True)

        single_cav_list = scenario_manager.create_vehicle_manager(
            application=['single'],
            map_helper=map_api.spawn_helper_2lanefree)

        traffic_manager, bg_veh_list = scenario_manager.create_traffic_carla()

        eval_manager = EvaluationManager(
            scenario_manager.cav_world,
            script_name='agent_single_llm_tool_demo',
            current_time=scenario_params['current_time'])

        csv_path = os.path.join(
            'opencda_output',
            'agent_single_llm_tool_demo',
            'agent_single_llm_tool_demo.csv')
        recorder = LLMExperimentRecorder(csv_path)

        spectator = scenario_manager.world.get_spectator()
        max_steps = int(scenario_params['scenario'].get('max_steps', 1000))

        for step in range(max_steps):
            scenario_manager.tick()

            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=70),
                carla.Rotation(pitch=-90)))

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)

                if i == 0 and recorder is not None:
                    recorder.record_step(step, single_cav, control)

            if step % 50 == 0:
                loc = single_cav_list[0].vehicle.get_location()
                print('[agent_single_llm_tool_demo] step=%d ego=(%.2f, %.2f)' %
                      (step, loc.x, loc.y))

    finally:
        if recorder is not None:
            recorder.save()

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
