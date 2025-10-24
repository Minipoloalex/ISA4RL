from dataclasses import dataclass
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import List
from utils import CONFIG_FILE

@dataclass
class RunConfig:
    id: int
    folder_name: str
    env: gym.Env
    model: BaseAlgorithm
    timesteps: int
    seed: int
