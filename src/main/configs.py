from dataclasses import dataclass, field
from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

EnvFactory = Callable[[], VecEnv]
ModelFactory = Callable[[VecEnv], BaseAlgorithm]
EvalEnvFactory = Callable[[], gym.Env]


@dataclass
class InstanceConfig:
    id: int
    orig_id_env_config: int
    id_env_config: int
    id_obs_config: int
    make_eval_env: EvalEnvFactory
    eval_seed: int
    instance_folder_name: str
    _eval_env: Optional[gym.Env] = field(default=None, init=False, repr=False)

    def ensure_eval_env(self) -> gym.Env:
        """Instantiate a single-environment instance for evaluation."""
        if self._eval_env is None:
            self._eval_env = self.make_eval_env()
        return self._eval_env

    def close(self) -> None:
        """Close any cached environment and drop references to heavy objects."""
        if self._eval_env is not None:
            try:
                self._eval_env.close()
            finally:
                self._eval_env = None

@dataclass
class RunConfig(InstanceConfig):
    train_id: int
    id_algo_config: int
    make_model: ModelFactory
    make_env: EnvFactory
    timesteps: int
    train_seed: int
    train_folder_name: str
    eval_freq: int
    n_eval_episodes: int
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
        super().close()
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

        self._model = None
