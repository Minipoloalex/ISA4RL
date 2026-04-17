from stable_baselines3 import A2C, DQN, PPO, SAC, TD3, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import gymnasium as gym

from typing import Optional, Dict, Any, Callable

from common.file_utils import *
from .general_utils import _coerce_numeric

AlgorithmName = str

ALGORITHM_MAP: Dict[AlgorithmName, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "dqn": DQN,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}

def make_env_helper(env_id: str, env_config: Dict[str, Any]) -> gym.Env:
    env_kwargs = {"config": env_config.copy()}
    # Nones are required to allow using **env_kwargs
    return gym.make(env_id, None, None, **env_kwargs)

def make_vec_env_helper(
    env_id: str,
    env_config: Dict[str, Any],
    env_cnt: int,
    vec_cls: type[DummyVecEnv] | type[SubprocVecEnv],
    vec_kwargs: Dict[str, Any] | None = None,
    monitor_dir: str | None = None,
) -> VecEnv:
    env_kwargs = {"config": env_config.copy(), "render_mode": None}
    return make_vec_env(
        env_id,
        n_envs=env_cnt,
        vec_env_cls=vec_cls,
        env_kwargs=env_kwargs,
        vec_env_kwargs=vec_kwargs,
        monitor_dir=monitor_dir,
    )

def make_env_vec_normalize(builder: Callable[[], VecEnv], vec_normalize_path: Path, training: bool, **kwargs) -> VecEnv:
    if nonempty_file_in(vec_normalize_path):
        env = VecNormalize.load(str(vec_normalize_path), builder())
        env.training = training
        if not env.training:
            env.norm_reward = False # handled outside already but also here for good measure
        return env
    return VecNormalize(builder(), training=training, **kwargs)

def make_model_helper(
    env: VecEnv,
    *,
    algo_cls: type[BaseAlgorithm],
    folder_name: str,
    model_path: Path,
    policy_params: Dict[str, Any],
    device: str,
) -> BaseAlgorithm:
    if nonempty_file_in(model_path):
        custom_objects = {}
        lr = policy_params.get("learning_rate")
        if lr is not None and type(lr) is not str:
            custom_objects["learning_rate"] = lr
        model = algo_cls.load(str(model_path), env=env, device=device, custom_objects=custom_objects)
        model.tensorboard_log = folder_name
        return model
    return algo_cls(env=env, tensorboard_log=folder_name, device=device, **policy_params)

def get_env_id(env: gym.Env):
    assert(env.spec is not None)
    return env.spec.id

def map_algo_name_to_class(algo_name: str) -> type[BaseAlgorithm]:
    algo_cls = ALGORITHM_MAP.get(algo_name.lower())
    if algo_cls is None:
        raise KeyError(
            f"Unknown algorithm '{algo_name}'. Expected one of {sorted(ALGORITHM_MAP)}."
        )
    return algo_cls

def load_model(model_path: Path, algo_name: str) -> BaseAlgorithm:
    algo_cls = map_algo_name_to_class(algo_name)
    return algo_cls.load(str(model_path))

def _parse_policy_kwargs(raw_value: Any) -> Any:
    """Convert string-encoded policy kwargs into a Python object."""
    if not isinstance(raw_value, str):
        return raw_value
    allowed_names = {
        "RMSpropTFLike": RMSpropTFLike,
        "dict": dict,
    }
    try:
        return eval(raw_value, {"__builtins__": {}}, allowed_names)
    except Exception as exc:  # pragma: no cover - config errors surface at runtime
        raise ValueError(f"Failed to parse policy_kwargs='{raw_value}': {exc}") from exc

def _linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func

def _resolve_schedule_placeholders(value: Any) -> Any:
    """Recursively replace lin_* placeholders with schedule callables."""
    if isinstance(value, str) and value.startswith("lin_"):
        parts = value.split("_")
        assert(len(parts) == 2)
        numeric_value = _coerce_numeric(parts[1])
        if numeric_value is None:
            raise ValueError(f"Invalid linear schedule specification: '{value}'")
        return _linear_schedule(numeric_value)
    return value

def unwrap_first_env(vec_env: VecEnv) -> Optional[gym.Env]:
    """Return the first underlying gym.Env from a VecEnv when accessible.

    Subproc-based VecEnvs keep environments in separate processes, making the
    raw env objects unpicklable across process boundaries. In that case we fall
    back to returning ``None`` and callers are expected to query attributes via
    ``VecEnv.get_attr`` instead.
    """
    current: VecEnv = vec_env
    # Unwrap nested VecEnv wrappers if present (e.g. VecMonitor -> VecNormalize -> VecEnvBase)
    for _ in range(10):
        envs = getattr(current, "envs", None)
        if envs:
            return envs[0].unwrapped
        next_vec = getattr(current, "venv", None)
        if next_vec is None:
            break
        current = next_vec
    return None

def find_vec_normalize(env: VecEnv) -> VecNormalize | None:
    """Recursively search for VecNormalize in the wrapper stack."""
    current_env = env
    while current_env is not None:
        if isinstance(current_env, VecNormalize):
            return current_env
        if hasattr(current_env, "venv"):
            current_env = current_env.venv
        else:
            break
    return None
