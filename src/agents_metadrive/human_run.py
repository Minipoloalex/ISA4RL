from panda3d.core import loadPrcFileData
loadPrcFileData("", "notify-level-linmath error")

from metadrive.envs.metadrive_env import MetaDriveEnv

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

import numpy as np

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config

# Change this string to test different map combinations!
MAP_CONFIG = "rORY"
MAP_CONFIG={
    BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
    BaseMap.GENERATE_CONFIG: MAP_CONFIG,  # it can be a file path / block num / block ID sequence
    BaseMap.LANE_WIDTH: 3.5,
    BaseMap.LANE_NUM: 4,
    "exit_length": 50,
    "start_position": [0, 0],
}

# Control settings
MANUAL_CONTROL = True    # Set to True to drive with W/A/S/D
TRAFFIC_DENSITY = 0.1    # Set to 0.0 to test just the road, >0.0 to add cars
NUM_STEPS = 500000         # How long the simulation runs before auto-closing

# ==========================================

def run_map_test():
    # Setup the environment configuration
    config = {
        "map_config": MAP_CONFIG,
        "manual_control": MANUAL_CONTROL,
        "use_render": True,
        "traffic_density": TRAFFIC_DENSITY,
        "num_scenarios": 10,
        "accident_prob": 1,
    }

    env = MetaDriveEnv(config)

    try:
        obs, info = env.reset()
        print(f"\n--- Successfully loaded map: '{MAP_CONFIG}' ---")
        if MANUAL_CONTROL:
            print("Manual control is ON. Use W/A/S/D to drive and test the blocks.")
        else:
            print("Manual control is OFF. The vehicle will take random actions.")

        # Simulation loop
        for step in range(NUM_STEPS):
            # If manual control is on, the action passed to step() is ignored 
            # and keyboard input is used instead.
            action = env.action_space.sample() if not MANUAL_CONTROL else [0, 0]

            obs, reward, terminated, truncated, info = env.step(action)

            # Reset the environment if the car crashes or finishes the map
            if terminated or truncated:
                print("Episode ended (crash or destination reached). Resetting...")
                env.reset()
                
    finally:
        env.close()
        print("Simulation closed.")

if __name__ == "__main__":
    run_map_test()
