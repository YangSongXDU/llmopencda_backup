# -*- coding: utf-8 -*-
"""
Scenario testing: Town06 multi-lane traffic demo for the LLM Sensor-as-Tools
Agent.
"""

import os
import subprocess
import sys

import carla

import opencda.scenario_testing.utils.sim_api as sim_api

from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time
from opencda.customize.core.common.llm_experiment_recorder import \
    LLMExperimentRecorder


def _run_postprocess(csv_path, config):
    """Generate the summary and figures after the CSV has been closed."""
    config = config or {}
    if not bool(config.get('enabled', False)):
        return

    project_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.abspath(os.path.join(
        project_root,
        str(config.get(
            'output_dir', 'results/agent_single_llm_tool_demo_2'))))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    commands = [
        [
            sys.executable,
            os.path.join(project_root, 'scripts',
                         'analyze_agent_tool_selection_csv.py'),
            '--csv', os.path.abspath(csv_path),
            '--out-json', os.path.join(output_dir, 'summary.json'),
            '--out-md', os.path.join(output_dir, 'summary.md'),
            '--quiet'
        ],
        [
            sys.executable,
            os.path.join(project_root, 'scripts',
                         'plot_agent_tool_selection_results.py'),
            '--csv', os.path.abspath(csv_path),
            '--out-dir', output_dir
        ]
    ]
    for command in commands:
        try:
            return_code = subprocess.call(command, cwd=project_root)
            if return_code != 0:
                print('[agent_single_llm_tool_demo_2] postprocess failed '
                      'with exit code %d: %s' % (
                          return_code, ' '.join(command)))
        except Exception as exc:
            print('[agent_single_llm_tool_demo_2] postprocess error: %s' % exc)


def _destination_location(scenario_params):
    destination = scenario_params['scenario']['single_cav_list'][0]['destination']
    return carla.Location(
        x=float(destination[0]),
        y=float(destination[1]),
        z=float(destination[2]))


def run_scenario(opt, scenario_params):
    scenario_manager = None
    eval_manager = None
    single_cav_list = []
    bg_veh_list = []
    recorder = None
    csv_path = None

    try:
        scenario_params = add_current_time(scenario_params)
        cav_world = CavWorld(opt.apply_ml)
        scenario_manager = sim_api.ScenarioManager(
            scenario_params,
            opt.apply_ml,
            opt.version,
            town='Town06',
            cav_world=cav_world)

        if opt.record:
            scenario_manager.client.start_recorder(
                'agent_single_llm_tool_demo_2.log', True)

        single_cav_list = scenario_manager.create_vehicle_manager(
            application=['single'])
        traffic_manager, bg_veh_list = scenario_manager.create_traffic_carla()

        eval_manager = EvaluationManager(
            scenario_manager.cav_world,
            script_name='agent_single_llm_tool_demo_2',
            current_time=scenario_params['current_time'])

        csv_path = os.path.join(
            'opencda_output',
            'agent_single_llm_tool_demo_2',
            'agent_single_llm_tool_demo_2.csv')
        recorder = LLMExperimentRecorder(
            csv_path,
            scenario_params['scenario'].get('resource_monitor', {}))

        llm_sensor_agent = getattr(
            single_cav_list[0].agent, 'llm_sensor_agent', None)
        if llm_sensor_agent is not None and \
                getattr(llm_sensor_agent, 'preflight_enabled', False):
            preflight = llm_sensor_agent.preflight()
            print('[agent_single_llm_tool_demo_2] LLM preflight '
                  'provider=%s model=%s available=%s fallback=%s '
                  'status=%s latency_ms=%.1f error=%s' % (
                      preflight.get('provider', ''),
                      preflight.get('model', ''),
                      preflight.get('provider_available', False),
                      preflight.get('fallback_used', False),
                      preflight.get('http_status', 0),
                      preflight.get('latency_ms', 0.0),
                      preflight.get('error', '')))

        spectator = scenario_manager.world.get_spectator()
        destination = _destination_location(scenario_params)
        max_steps = int(scenario_params['scenario'].get('max_steps', 2400))
        min_run_steps = int(scenario_params['scenario'].get('min_run_steps', 120))
        goal_radius = float(scenario_params['scenario'].get('goal_radius', 12.0))
        if max_steps <= 0:
            max_steps = 2400
        print('[agent_single_llm_tool_demo_2] max_steps=%d goal_radius=%.1f' %
              (max_steps, goal_radius))
        last_overtake_state = None

        for step in range(max_steps):
            scenario_manager.tick()

            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=85),
                carla.Rotation(pitch=-90)))

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)

                if i == 0 and recorder is not None:
                    recorder.record_step(step, single_cav, control)

            ego_loc = single_cav_list[0].vehicle.get_location()
            ego_agent = single_cav_list[0].agent
            current_state = getattr(ego_agent, 'overtake_state', '')
            if current_state != last_overtake_state:
                print('[agent_single_llm_tool_demo_2] step=%d state=%s '
                      'attempts=%d completed=%d ego=(%.2f, %.2f)' % (
                          step,
                          current_state,
                          int(getattr(ego_agent, 'overtake_attempt_count', 0)),
                          int(getattr(ego_agent, 'completed_overtake_count', 0)),
                          ego_loc.x,
                          ego_loc.y))
                last_overtake_state = current_state
            if step % 50 == 0:
                print('[agent_single_llm_tool_demo_2] step=%d ego=(%.2f, %.2f) '
                      'state=%s completed=%d' % (
                          step,
                          ego_loc.x,
                          ego_loc.y,
                          current_state,
                          int(getattr(
                              ego_agent, 'completed_overtake_count', 0))))

            if step > min_run_steps and ego_loc.distance(destination) < goal_radius:
                print('[agent_single_llm_tool_demo_2] ego reached goal region '
                      'at step=%d distance=%.2f' %
                      (step, ego_loc.distance(destination)))
                break

    finally:
        if recorder is not None:
            recorder.save()
            _run_postprocess(
                csv_path,
                scenario_params['scenario'].get('postprocess', {}))

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
