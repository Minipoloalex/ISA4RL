from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from functools import partial

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv

from configs import RunConfig, InstanceConfig
from common.file_utils import *
from common.config_utils import CONFIG, map_to_train_id, get_all_instance_configs
from main.utils.sb3_utils import (
    _resolve_schedule_placeholders,
    map_algo_name_to_class,
    _parse_policy_kwargs,
    _make_env,
    _make_vec_env,
    _make_model,
)

# Mostly refer to the environment
_DISCARD_POLICY_PARAMS = ["n_envs", "algo", "env_wrapper", "frame_stack", "normalize", "id"]

def load_all_run_configs(get_configs: Callable[[], List[CONFIG]]) -> List[RunConfig]:
    """get_configs should be get_all_run_configs or get_all_eval_configs"""
    configs = get_configs()
    run_configs: List[RunConfig] = []
    for config in configs:
        env_config : CONFIG = config["env_config"]
        obs_config : CONFIG = config["obs_config"]
        algo_config: CONFIG = config["algo_config"]

        id: int = config["id"]
        id_env: int = env_config["id"]
        orig_id_env: int = env_config["orig_id"]    # TODO: remove the other one (with .get())
        id_obs: int = obs_config["id"]
        id_algo: int = algo_config["id"]
        train_id: int = map_to_train_id(orig_id_env, id_obs, id_algo)

        # Depend on environment (some environments may have very large episodes)
        eval_freq: int = env_config["eval_freq"]
        n_eval_episodes: int = env_config["n_eval_episodes"]
        train_timesteps: int = env_config["train_timesteps"]

        train_folder_name: str = str(BASE_OUTPUT_PATH / TRAIN_FOLDER / str(train_id))
        instance_folder_name: str = str(BASE_OUTPUT_PATH / METAFEATURES_FOLDER / f"ENV{id_env}_OBS{id_obs}")

        if "observation_shape" in obs_config:
            obs_config["observation_shape"] = tuple(obs_config["observation_shape"])

        env_id: str = env_config["env_id"]
        eval_seed: int = env_config["eval_seed"]

        env_config: CONFIG = env_config["config"]
        env_config["observation"] = obs_config

        algo_name: str = algo_config["algo"]
        n_envs = algo_config.get("n_envs", 1)
        policy = algo_config["policy"]
        for key in _DISCARD_POLICY_PARAMS:
            algo_config.pop(key, None)

        for param in algo_config.keys():
            algo_config[param] = _resolve_schedule_placeholders(algo_config[param])

        policy_params: Dict[str, Any] = algo_config
        policy_kwargs = policy_params.get("policy_kwargs")
        if policy_kwargs is not None:
            policy_params["policy_kwargs"] = _parse_policy_kwargs(policy_kwargs)
        algo_cls = map_algo_name_to_class(algo_name)

        vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
        vec_env_kwargs = None if n_envs == 1 else {"start_method": "spawn"}
        device = "cuda" if policy == "CnnPolicy" else "cpu"

        run_configs.append(
            RunConfig(
                id=id,
                train_id=train_id,
                id_env_config=id_env,
                orig_id_env_config=orig_id_env,
                id_obs_config=id_obs,
                id_algo_config=id_algo,
                instance_folder_name=instance_folder_name,
                train_folder_name=train_folder_name,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                make_env=partial(_make_env, env_id, env_config),
                make_vec_env=partial(_make_vec_env, env_id, env_config, n_envs, vec_env_cls, vec_env_kwargs),
                make_model=partial(
                    _make_model,
                    algo_cls=algo_cls,
                    folder_name=train_folder_name,
                    policy_params=policy_params,
                    device=device,
                ),
                timesteps=train_timesteps,
                train_seed=0,
                eval_seed=eval_seed,
            )
        )
    return run_configs

def load_all_instance_configs() -> List[InstanceConfig]:
    configs = get_all_instance_configs()
    instance_configs: List[InstanceConfig] = []
    for config in configs:
        env_config : Dict[str, Any] = config["env_config"]
        obs_config : Dict[str, Any] = config["obs_config"]

        id: int = config["id"]
        id_env: int = env_config["id"]
        orig_id_env: int = env_config["orig_id"]
        id_obs: int = obs_config["id"]
        instance_folder_name: str = str(BASE_OUTPUT_PATH / METAFEATURES_FOLDER / f"ENV{id_env}_OBS{id_obs}")

        if "observation_shape" in obs_config:
            obs_config["observation_shape"] = tuple(obs_config["observation_shape"])

        env_id: str = env_config["env_id"]
        env_config: Dict[str, Any] = env_config["config"]
        env_config["observation"] = obs_config

        instance_configs.append(
            InstanceConfig(
                id=id,
                id_env_config=id_env,
                orig_id_env_config=orig_id_env,
                id_obs_config=id_obs,
                make_env=partial(_make_env, env_id, env_config),
                eval_seed=1,
                instance_folder_name=instance_folder_name,
            )
        )
    return instance_configs

def is_trained(config: RunConfig) -> bool:
    """Trained iff training metadata artifact exists."""
    return nonempty_file_in(Path(config.train_folder_name) / TRAINING_METADATA_FILE)

def is_evaluated(config: RunConfig) -> bool:
    """Evaluated iff evaluation results artifact exists."""
    return nonempty_file_in(
        Path(config.train_folder_name)
        / EVALUATION_RESULTS_BASE_PATH
        / EVALUATION_RESULTS_FILE(config.eval_seed)
    )

def is_extracted(config: InstanceConfig) -> bool:
    """Extracted iff metafeatures result artifact exists."""
    return nonempty_file_in(Path(config.instance_folder_name) / METAFEATURES_RESULTS_FILE)

def annotate_ids(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            **d,
            "id": id,
        }
        for id, d in enumerate(lst)
    ]
