from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path
from functools import partial
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv, VecNormalize

from common.file_utils import *
from common.config_utils import CONFIG, get_algo_id, get_instance_id, save_algo_config, save_instance_config
from utils.sb3_utils import (
    _resolve_schedule_placeholders,
    map_algo_name_to_class,
    _parse_policy_kwargs,
    make_env_helper,
    make_vec_env_helper,
    make_env_vec_normalize,
    make_model_helper,
)

# Mostly refer to the environment
_DISCARD_POLICY_PARAMS = ["algo", "action_space", "n_envs", "env_wrapper", "frame_stack", "normalize", "id"]

EnvFactory = Callable[[], gym.Env]
VecEnvFactory = Callable[[], VecEnv]
ModelFactory = Callable[[VecEnv], BaseAlgorithm]

@dataclass
class InstanceConfig:
    make_test_env: EnvFactory
    n_test_episodes: int
    instance_folder_path: Path

    # Only used for metafeatures extraction (no normalization)
    _test_env: Optional[gym.Env] = field(default=None, init=False, repr=False)

    @classmethod
    def from_dict(cls, env_name: str, config_dict: CONFIG, base_results_path: Path = BASE_RESULTS_PATH) -> "InstanceConfig":
        env_config = config_dict["env_config"].copy()
        obs_config = config_dict["obs_config"].copy()

        base_env_path = RESULTS_ENV_FOLDER_PATH(base_results_path, env_name)
        instance_config_id = get_instance_id(base_env_path, env_config, obs_config)
        instance_folder_path = RESULTS_INSTANCE_FOLDER_PATH(base_env_path, instance_config_id)
        ensure_dir(instance_folder_path)

        if "observation_shape" in obs_config:
            obs_config["observation_shape"] = tuple(obs_config["observation_shape"])

        env_id = env_config["env_id"]
        env_kwargs = env_config["config"]
        env_kwargs["observation"] = obs_config

        return cls(
            make_test_env=partial(make_env_helper, env_id, env_kwargs),
            n_test_episodes=env_config["n_test_episodes"],
            instance_folder_path=instance_folder_path,
        )

    def ensure_test_env(self) -> gym.Env:
        """Instantiate a single-environment instance for evaluation."""
        if self._test_env is None:
            self._test_env = self.make_test_env()
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
    make_train_env: VecEnvFactory
    make_eval_env: VecEnvFactory
    timesteps: int
    train_folder_path: Path
    eval_freq: int
    n_eval_episodes: int
    _train_env: Optional[VecEnv] = field(default=None, init=False, repr=False)
    _eval_env: Optional[VecEnv] = field(default=None, init=False, repr=False)
    _model: Optional[BaseAlgorithm] = field(default=None, init=False, repr=False)

    @classmethod
    def from_dict(
        cls,
        env_name: str,
        config_dict: Dict[str, Any],
        base_results_path: Path = BASE_RESULTS_PATH,
        use_best_model: bool = True,
    ) -> "TrainConfig":
        """
        use_best_model: only useful for evaluating (loads best model instead of model saved last)
        """
        env_config = config_dict["env_config"].copy()
        obs_config = config_dict["obs_config"].copy()
        algo_config = config_dict["algo_config"].copy()

        base_env_path = RESULTS_ENV_FOLDER_PATH(base_results_path, env_name)
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

        env_id = env_config["env_id"]
        env_kwargs = env_config["config"]
        env_kwargs["observation"] = obs_config

        algo_name = algo_config.pop("algo")
        use_vec_normalize = algo_config.pop("normalize", False)
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

        model_file = BEST_MODEL_FILE if use_best_model else MODEL_FILE
        vec_normalize_file = BEST_VEC_NORMALIZE_FILE if use_best_model else VEC_NORMALIZE_FILE
        load_model_path = train_algo_folder_path / model_file # in case it's evaluation
        load_vec_normalize_path = train_algo_folder_path / vec_normalize_file

        # Only normalize rewards for parking
        normalize_reward = use_vec_normalize and env_id == "parking-v0"

        train_vec_env_builder = partial(
            make_vec_env_helper, env_id, env_kwargs, n_envs, vec_env_cls, vec_env_kwargs, monitor_dir=str(train_algo_folder_path),
        )
        eval_vec_env_builder = partial(
            make_vec_env_helper, env_id, env_kwargs, 1, DummyVecEnv, monitor_dir=str(train_algo_folder_path),
        )

        if use_vec_normalize:
            common_kwargs = {
                "norm_obs": True,
                "clip_obs": 10,
                "clip_reward": 10,  # only used if norm_reward is True
            }
            train_vec_env_builder = partial(
                make_env_vec_normalize, train_vec_env_builder, load_vec_normalize_path,
                training=True, norm_reward=normalize_reward,**common_kwargs,
            )
            eval_vec_env_builder = partial(
                make_env_vec_normalize, eval_vec_env_builder, load_vec_normalize_path,
                training=False, norm_reward=False, **common_kwargs,
            )
        return cls(
            # make_test_env not used in TrainConfig
            make_test_env=partial(make_env_helper, env_id, env_kwargs),

            make_eval_env=eval_vec_env_builder,
            n_test_episodes=env_config["n_test_episodes"],
            instance_folder_path=instance_folder_path,
            make_model=partial(
                make_model_helper,
                algo_cls=algo_cls,
                folder_name=str(train_algo_folder_path),
                model_path=load_model_path,
                policy_params=policy_params,
                device=device,
            ),
            make_train_env=train_vec_env_builder,
            timesteps=env_config["train_timesteps"],
            train_folder_path=train_algo_folder_path,
            eval_freq=env_config["eval_freq"] // n_envs,
            n_eval_episodes=env_config["n_eval_episodes"],
        )

    def ensure_train_env(self) -> VecEnv:
        """Instantiate the environment lazily and cache it for reuse."""
        if self._train_env is None:
            self._train_env = self.make_train_env()
        return self._train_env

    def ensure_eval_env(self) -> VecEnv:
        if self._eval_env is None:
            self._eval_env = self.make_eval_env()
        return self._eval_env

    def ensure_test_env(self) -> gym.Env:
        # Not supposed to use test_env within TrainConfig
        raise NotImplementedError

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
