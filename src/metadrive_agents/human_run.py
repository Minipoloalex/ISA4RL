from copy import deepcopy

from panda3d.core import loadPrcFileData
from metadrive.component.sensors.rgb_camera import RGBCamera
# from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.my_metadrive_env import MyMetaDriveEnv
from metadrive.envs.metadrive_env import METADRIVE_DEFAULT_CONFIG as _METADRIVE_DEFAULT_CONFIG

loadPrcFileData("", "notify-level-linmath error")

# ==========================================
#              CONFIGURATION
# ==========================================



# Define your map using block characters. 
# Common blocks:
# 'S' = Straight
# 'C' = Curve (Circular)
# 'T' = T-Intersection
# 'X' = Crossroad
# 'O' = Roundabout
# 'R' = Ramp (entry/exit)
# 'r' = Ramp (straight)

import copy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union
import logging

import numpy as np

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config

logger = logging.getLogger(__name__)

# Change this string to test different map combinations
MAP_CONFIG = "SC"
MAP_CONFIG={
    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
    BaseMap.GENERATE_CONFIG: MAP_CONFIG,
    BaseMap.LANE_WIDTH: 3.5,
    BaseMap.LANE_NUM: 5,
    "exit_length": 50,
    "start_position": [0, 0],
}

# Control settings
MANUAL_CONTROL = True    # Set to True to drive with W/A/S/D
TRAFFIC_DENSITY = 0.1    # Set to 0.0 to test just the road, >0.0 to add cars
NUM_STEPS = 500000         # How long the simulation runs before auto-closing

# ==========================================


def run_map_test():
    INTERSECTION_PROB_DIST = {"StdInterSection": 1/3, "StdTInterSection": 1/3, "Roundabout": 1/3}
    # MapGenerateMethod.BIG_BLOCK_NUM or # MapGenerateMethod.BIG_BLOCK_SEQUENCE

    # config = deepcopy(_METADRIVE_DEFAULT_CONFIG)
    config = {}
    config.update({
        "store_map": False,
        "traffic_mode": "respawn",
        "manual_control": MANUAL_CONTROL,
        "use_render": True,
        "traffic_density": TRAFFIC_DENSITY,
        "image_observation": False,
        "num_scenarios": 1000000000000,
        "accident_prob": 0,
        "random_traffic": False,
        # "image_on_cuda": True,
        # "block_dist_config": PGBlockDistConfig(INTERSECTION_PROB_DIST),
        "sensors": dict(rgb_camera=(RGBCamera, 200, 100)),
        "interface_panel": ["rgb_camera", "dashboard"],
        "stack_size": 3,
        "random_agent_model": False,
        "horizon": 800,
    })
    # config["map_config"].update({
    #     BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
    #     BaseMap.GENERATE_CONFIG: 1,
    # })
    config["map_config"] = MAP_CONFIG

    # config["vehicle_config"] = {
    #     "vehicle_model": "m",
    # }
    env = MyMetaDriveEnv(config)

    try:
        obs, info = env.reset()
        logger.info(f"\n--- Successfully loaded map' ---")
        if MANUAL_CONTROL:
            logger.info("Manual control is ON. Use W/A/S/D to drive and test the blocks.")
        else:
            logger.info("Manual control is OFF. The vehicle will take random actions.")

        # Simulation loop
        for step in range(NUM_STEPS):
            # If manual control is on, the action passed to step() is ignored 
            # and keyboard input is used instead.
            action = env.action_space.sample() if not MANUAL_CONTROL else [0, 0]

            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(reward)

            # Reset the environment if the car crashes or finishes the map
            if terminated or truncated:
                logger.info("Episode ended (crash or destination reached). Resetting...")
                env.reset()
                
    finally:
        env.close()
        logger.info("Simulation closed.")

if __name__ == "__main__":
    run_map_test()
