from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path
from functools import partial
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv

from common.file_utils import *
from common.config_utils import CONFIG, get_algo_id, get_instance_id, save_algo_config, save_instance_config
from utils.sb3_utils import (
    _resolve_schedule_placeholders,
    map_algo_name_to_class,
    _parse_policy_kwargs,
    _make_env,
    _make_vec_env,
    _make_model,
)

# Mostly refer to the environment
_DISCARD_POLICY_PARAMS = ["n_envs", "algo", "env_wrapper", "frame_stack", "normalize", "id"]

EnvFactory = Callable[[], gym.Env]
VecEnvFactory = Callable[[], VecEnv]
ModelFactory = Callable[[VecEnv], BaseAlgorithm]

@dataclass
class InstanceConfig:
    make_env: EnvFactory
    n_test_episodes: int
    instance_folder_path: Path
    _test_env: Optional[gym.Env] = field(default=None, init=False, repr=False)

    @classmethod
    def from_dict(cls, env_name: str, config_dict: CONFIG) -> "InstanceConfig":
        env_config = config_dict["env_config"].copy()
        obs_config = config_dict["obs_config"].copy()

        base_env_path = RESULTS_ENV_FOLDER_PATH(env_name)
        instance_config_id = get_instance_id(base_env_path, env_config, obs_config)
        instance_folder_path = RESULTS_INSTANCE_FOLDER_PATH(base_env_path, instance_config_id)
        ensure_dir(instance_folder_path)

        if "observation_shape" in obs_config:
            obs_config["observation_shape"] = tuple(obs_config["observation_shape"])

        env_kwargs = env_config["config"]
        env_kwargs["observation"] = obs_config

        return cls(
            make_env=partial(_make_env, env_name, env_kwargs),
            n_test_episodes=env_config["n_test_episodes"],
            instance_folder_path=instance_folder_path,
        )

    def ensure_test_env(self) -> gym.Env:
        """Instantiate a single-environment instance for evaluation."""
        if self._test_env is None:
            self._test_env = self.make_env()
        return self._test_env

    def close(self) -> None:
        """Close any cached environment and drop references to heavy objects."""
        if self._test_env is not None:
            try:
                self._test_env.close()
            finally:
                self._test_env = None

@dataclass
class TrainConfig(InstanceConfig):
    make_model: ModelFactory
    make_vec_env: VecEnvFactory
    timesteps: int
    train_folder_path: Path
    eval_freq: int
    n_eval_episodes: int
    _train_env: Optional[VecEnv] = field(default=None, init=False, repr=False)
    _eval_env: Optional[VecEnv] = field(default=None, init=False, repr=False)
    _model: Optional[BaseAlgorithm] = field(default=None, init=False, repr=False)

    @classmethod
    def from_dict(cls, env_name: str, config_dict: Dict[str, Any]) -> "TrainConfig":
        env_config = config_dict["env_config"].copy()
        obs_config = config_dict["obs_config"].copy()
        algo_config = config_dict["algo_config"].copy()

        base_env_path = RESULTS_ENV_FOLDER_PATH(env_name)
        instance_config_id = get_instance_id(base_env_path, env_config, obs_config)
        instance_folder_path = RESULTS_INSTANCE_FOLDER_PATH(base_env_path, instance_config_id)
        ensure_dir(instance_folder_path)
        save_instance_config(instance_folder_path, env_config, obs_config)

        train_folder_path = RESULTS_TRAIN_FOLDER_PATH(instance_folder_path)
        ensure_dir(train_folder_path)

        algo_config_id = get_algo_id(instance_folder_path, algo_config)
        train_algo_folder_path = RESULTS_TRAIN_ALGO_FOLDER_PATH(train_folder_path, algo_config_id)
        ensure_dir(train_algo_folder_path)
        save_algo_config(train_algo_folder_path, algo_config)

        if "observation_shape" in obs_config:
            obs_config["observation_shape"] = tuple(obs_config["observation_shape"])

        env_kwargs = env_config["config"]
        env_kwargs["observation"] = obs_config

        algo_name = algo_config.pop("algo")
        n_envs = algo_config.pop("n_envs", 1)
        policy = algo_config["policy"]

        for key in _DISCARD_POLICY_PARAMS:
            algo_config.pop(key, None)

        for param, value in algo_config.items():
            algo_config[param] = _resolve_schedule_placeholders(value)

        policy_params = algo_config
        if "policy_kwargs" in policy_params and policy_params["policy_kwargs"] is not None:
            policy_params["policy_kwargs"] = _parse_policy_kwargs(policy_params["policy_kwargs"])

        algo_cls = map_algo_name_to_class(algo_name)
        vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
        vec_env_kwargs = None if n_envs == 1 else {"start_method": "spawn"}
        device = "cuda" if policy == "CnnPolicy" else "cpu"

        return cls(
            make_env=partial(_make_env, env_name, env_kwargs),
            n_test_episodes=env_config["n_test_episodes"],
            instance_folder_path=instance_folder_path,
            make_model=partial(
                _make_model,
                algo_cls=algo_cls,
                folder_name=str(train_algo_folder_path),
                policy_params=policy_params,
                device=device,
            ),
            make_vec_env=partial(_make_vec_env, env_name, env_kwargs, n_envs, vec_env_cls, vec_env_kwargs),
            timesteps=env_config["train_timesteps"],
            train_folder_path=train_algo_folder_path,
            eval_freq=env_config["eval_freq"],
            n_eval_episodes=env_config["n_eval_episodes"],
        )

    def ensure_train_env(self) -> VecEnv:
        """Instantiate the environment lazily and cache it for reuse."""
        if self._train_env is None:
            self._train_env = self.make_vec_env()
        return self._train_env

    def ensure_eval_env(self) -> VecEnv:
        if self._eval_env is None:
            self._eval_env = self.make_vec_env()
        return self._eval_env

    def ensure_model(self) -> BaseAlgorithm:
        """Instantiate the algorithm lazily and cache it alongside the env."""
        env = self.ensure_train_env()
        if self._model is None:
            self._model = self.make_model(env)
        return self._model

    def close(self) -> None:
        """Close any cached environment and drop references to heavy objects."""
        super().close()
        if self._train_env is not None:
            try:
                self._train_env.close()
            finally:
                self._train_env = None
        if self._eval_env is not None:
            try:
                self._eval_env.close()
            finally:
                self._eval_env = None
        self._model = None
