from dataclasses import dataclass, field
from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

EnvFactory = Callable[[], VecEnv]
ModelFactory = Callable[[VecEnv], BaseAlgorithm]


@dataclass
class RunConfig:
    id: int
    folder_name: str
    make_env: EnvFactory
    make_model: ModelFactory
    timesteps: int
    seed: int
    _env: Optional[VecEnv] = field(default=None, init=False, repr=False)
    _model: Optional[BaseAlgorithm] = field(default=None, init=False, repr=False)

    def ensure_env(self) -> VecEnv:
        """Instantiate the environment lazily and cache it for reuse."""
        if self._env is None:
            self._env = self.make_env()
        return self._env

    def ensure_model(self) -> BaseAlgorithm:
        """Instantiate the algorithm lazily and cache it alongside the env."""
        env = self.ensure_env()
        if self._model is None:
            self._model = self.make_model(env)
        return self._model

    def close(self) -> None:
        """Close any cached environment and drop references to heavy objects."""
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
        self._model = None
