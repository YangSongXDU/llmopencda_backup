# -*- coding: utf-8 -*-
"""
Basic class of CAV
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import uuid

from opencda.core.actuation.control_manager \
    import ControlManager
from opencda.core.application.platooning.platoon_behavior_agent\
    import PlatooningBehaviorAgent
from opencda.core.common.v2x_manager \
    import V2XManager
from opencda.core.sensing.localization.localization_manager \
    import LocalizationManager
from opencda.core.sensing.perception.perception_manager \
    import PerceptionManager
from opencda.core.safety.safety_manager import SafetyManager
from opencda.core.plan.behavior_agent \
    import BehaviorAgent
from opencda.customize.core.plan.llm_tool_behavior_agent \
    import LLMToolBehaviorAgent
from opencda.core.map.map_manager import MapManager
from opencda.core.common.data_dumper import DataDumper


class VehicleManager(object):
    """
    A class manager to embed different modules with vehicle together.
    """

    def __init__(
            self,
            vehicle,
            config_yaml,
            application,
            carla_map,
            cav_world,
            current_time='',
            data_dumping=False):

        # an unique uuid for this vehicle
        self.vid = str(uuid.uuid1())
        self.vehicle = vehicle
        self.carla_map = carla_map

        # retrieve the configure for different modules
        sensing_config = config_yaml['sensing']
        map_config = config_yaml['map_manager']
        behavior_config = config_yaml['behavior']
        control_config = config_yaml['controller']
        v2x_config = config_yaml['v2x']

        # v2x module
        self.v2x_manager = V2XManager(cav_world, v2x_config, self.vid)
        # localization module
        self.localizer = LocalizationManager(
            vehicle, sensing_config['localization'], carla_map)
        # perception module. This may still be used to spawn ego sensors and
        # visualize data. In self-perception mode, CARLA-server vehicle outputs
        # from detect() are sanitized before being passed to the LLM behavior.
        self.perception_manager = PerceptionManager(
            vehicle, sensing_config['perception'], cav_world,
            data_dumping)
        # map manager
        self.map_manager = MapManager(vehicle,
                                      carla_map,
                                      map_config)
        # safety manager
        self.safety_manager = SafetyManager(cav_world=cav_world,
                                            vehicle=vehicle,
                                            params=config_yaml['safety_manager'])
        # behavior agent
        self.agent = None
        if 'platooning' in application:
            platoon_config = config_yaml['platoon']
            self.agent = PlatooningBehaviorAgent(
                vehicle,
                self,
                self.v2x_manager,
                behavior_config,
                platoon_config,
                carla_map)
        else:
            if behavior_config.get('use_llm_tool_agent', False):
                self.agent = LLMToolBehaviorAgent(
                    vehicle, carla_map, behavior_config)
                self.agent.set_vehicle_manager(self)
            else:
                self.agent = BehaviorAgent(vehicle, carla_map, behavior_config)

        # Control module
        self.controller = ControlManager(control_config)

        if data_dumping:
            self.data_dumper = DataDumper(self.perception_manager,
                                          vehicle.id,
                                          save_time=current_time)
        else:
            self.data_dumper = None

        cav_world.update_vehicle_manager(self)

    @staticmethod
    def _sanitize_self_perception_objects(objects):
        """
        Remove CARLA-server vehicle objects while preserving other keys.

        OpenCDA SafetyManager still expects keys such as traffic_lights. We
        therefore only remove vehicles from the default perception output, while
        keeping non-vehicle information available for safety utilities.
        """
        sanitized = dict(objects) if isinstance(objects, dict) else {}
        sanitized['vehicles'] = []
        if 'traffic_lights' not in sanitized:
            sanitized['traffic_lights'] = []
        return sanitized

    def set_destination(
            self,
            start_location,
            end_location,
            clean=False,
            end_reset=True):
        """
        Set global route.
        """

        self.agent.set_destination(
            start_location, end_location, clean, end_reset)

    def update_info(self):
        """
        Call perception and localization module to
        retrieve surrounding info an ego position.
        """
        # localization
        self.localizer.localize()

        ego_pos = self.localizer.get_ego_pos()
        ego_spd = self.localizer.get_ego_spd()

        # object detection. This call also keeps OpenCDA sensor managers alive.
        # However, if the customized LLM agent is in self-perception mode,
        # the returned CARLA-server vehicle list must not be used for behavior.
        objects = self.perception_manager.detect(ego_pos)
        if getattr(self.agent, 'self_perception_only', False):
            method_objects = self._sanitize_self_perception_objects(objects)
        else:
            method_objects = objects

        # update the ego pose for map manager
        self.map_manager.update_information(ego_pos)

        # this is required by safety manager
        safety_input = {
            'ego_pos': ego_pos,
            'ego_speed': ego_spd,
            'objects': method_objects,
            'carla_map': self.carla_map,
            'world': self.vehicle.get_world(),
            'static_bev': self.map_manager.static_bev,
            'vis_bev': self.map_manager.vis_bev
        }
        self.safety_manager.update_info(safety_input)

        # update ego position and speed to v2x manager,
        # and then v2x manager will search the nearby cavs
        self.v2x_manager.update_info(ego_pos, ego_spd)

        self.agent.update_information(ego_pos, ego_spd, method_objects)
        # pass position and speed info to controller
        self.controller.update_info(ego_pos, ego_spd)

    def run_step(self, target_speed=None):
        """
        Execute one step of navigation.
        """
        # visualize the bev map if needed
        self.map_manager.run_step()
        target_speed, target_pos = self.agent.run_step(target_speed)
        control = self.controller.run_step(target_speed, target_pos)

        # dump data
        if self.data_dumper:
            self.data_dumper.run_step(self.perception_manager,
                                      self.localizer,
                                      self.agent)

        return control

    def destroy(self):
        """
        Destroy the actor vehicle
        """
        if hasattr(self.agent, 'destroy'):
            self.agent.destroy()
        self.perception_manager.destroy()
        self.localizer.destroy()
        self.vehicle.destroy()
        self.map_manager.destroy()
