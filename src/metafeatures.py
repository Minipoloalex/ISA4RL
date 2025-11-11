import hashlib
import math
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils import _flatten_obs

import numpy as np
import gymnasium as gym

from configs import InstanceConfig
from highway_env.vehicle.behavior import IDMVehicle

PolicyFn = Callable[[Any, Dict[str, Any]], Any]


def extract_metafeatures(config: InstanceConfig):
    env = config.ensure_eval_env()
    env_name = getattr(getattr(env, "spec", None), "id", None)
    horizon = _infer_horizon(env)
    base_seed = getattr(config, "eval_seed", None)
    env_features = _collect_env_features(env)

    random_probe = _run_probe(
        env=env,
        policy=_make_random_policy(env),
        episodes=10,
        max_steps=horizon,
        seed=base_seed,
        label="random_rollout",
    )
    idm_probe = _run_idm_probe(
        env=env,
        episodes=10,
        max_steps=horizon,
        seed=base_seed,
        label="idm_like",
    )
    config.close()

    idm_advantage = (
        idm_probe["mean_episode_return"] - random_probe["mean_episode_return"]
    )
    safety_delta = random_probe["collision_rate"] - idm_probe["collision_rate"]
    features: Dict[str, float] = {}
    features.update(env_features)
    features.update(_probe_to_features(random_probe, "random"))
    features.update(_probe_to_features(idm_probe, "idm"))
    features.update(_derived_features(random_probe, idm_probe, env_features))

    return {
        "instance_id": config.id,
        "env_config_id": config.id_env_config,
        "obs_config_id": config.id_obs_config,
        "env_name": env_name,
        "generated_at": time.time_ns(),
        "probes": {
            "random": random_probe,
            "idm_like": idm_probe,
        },
        "diagnostics": {
            "estimated_horizon": horizon,
            "idm_advantage": idm_advantage,
            "idm_safety_gain": safety_delta,
        },
        "features": features,
    }


def _make_random_policy(env: gym.Env) -> PolicyFn:
    action_space = env.action_space

    def policy(_: Any, __: Dict[str, Any]) -> Any:
        return action_space.sample()

    return policy


def _run_idm_probe(
    env: gym.Env,
    episodes: int,
    max_steps: Optional[int],
    seed: Optional[int],
    label: str,
) -> Dict[str, Any]:
    dummy_action = _default_idle_action(env)
    policy = _constant_policy(dummy_action)

    def reset_hook(target_env: gym.Env) -> None:
        _ensure_idm_vehicle(target_env)

    return _run_probe(
        env=env,
        policy=policy,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        label=label,
        reset_hook=reset_hook,
        track_actions=True,
        action_getter=_idm_action_getter,
    )


def _default_idle_action(env: gym.Env) -> Any:
    space = env.action_space
    spaces = gym.spaces
    if isinstance(space, spaces.Discrete):
        return 0
    return None


def _default_value_for_space(space: gym.spaces.Space) -> Any:
    spaces = gym.spaces
    if isinstance(space, spaces.Discrete):
        return 0
    if isinstance(space, spaces.MultiBinary):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, spaces.MultiDiscrete):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, spaces.Tuple):
        return tuple(_default_value_for_space(sub) for sub in space.spaces)
    return space.sample()


def _constant_policy(action: Any) -> PolicyFn:
    def policy(_: Any, __: Dict[str, Any]) -> Any:
        if isinstance(action, np.ndarray):
            return np.array(action, copy=True)
        if isinstance(action, (list, tuple)):
            return type(action)(action)
        return action

    return policy


def _base_env(env: gym.Env) -> gym.Env:
    return getattr(env, "unwrapped", env)


def _ensure_idm_vehicle(env: gym.Env) -> None:
    base_env = _base_env(env)
    vehicle = getattr(base_env, "vehicle", None)
    road = getattr(base_env, "road", None)

    idm_vehicle = IDMVehicle.create_from(vehicle)
    idm_vehicle.enable_lane_change = True
    idm_vehicle.timer = getattr(vehicle, "timer", getattr(idm_vehicle, "timer", 0.0))
    _copy_vehicle_attr(vehicle, idm_vehicle, "target_speed")
    _copy_vehicle_attr(vehicle, idm_vehicle, "target_speeds", deep_copy=True)
    _copy_vehicle_attr(vehicle, idm_vehicle, "index_to_speed")
    _copy_vehicle_attr(vehicle, idm_vehicle, "speed_index")

    base_env.vehicle = idm_vehicle

    for idx, existing in enumerate(road.vehicles):
        if existing is vehicle:
            road.vehicles[idx] = idm_vehicle
            break

    action_type = getattr(base_env, "action_type", None)
    if action_type is not None:
        action_type.controlled_vehicle = idm_vehicle


def _copy_vehicle_attr(
    src: Any, dst: Any, attr: str, *, deep_copy: bool = False
) -> None:
    if not hasattr(src, attr):
        return
    value = getattr(src, attr)
    if deep_copy:
        value = None if value is None else np.array(value, copy=True)
    setattr(dst, attr, value)


def _idm_action_getter(_: Any, env: gym.Env) -> Any:
    vehicle = getattr(_base_env(env), "vehicle", None)
    if vehicle is None:
        return None
    action = getattr(vehicle, "action", None)
    if isinstance(action, dict):
        normalized: Dict[str, Any] = {}
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                normalized[key] = np.array(value, copy=True)
            elif isinstance(value, (list, tuple)):
                normalized[key] = list(value)
            else:
                normalized[key] = value
        return normalized
    return action


def _run_probe(
    env: gym.Env,
    policy: PolicyFn,
    episodes: int,
    max_steps: Optional[int],
    seed: Optional[int],
    label: str,
    reset_hook: Optional[Callable[[gym.Env], None]] = None,
    track_actions: bool = True,
    action_getter: Optional[Callable[[Any, gym.Env], Any]] = None,
) -> Dict[str, Any]:
    returns: List[float] = []
    lengths: List[int] = []
    collisions = 0
    timeouts = 0
    steps_collected = 0

    speed_samples: List[float] = []
    obs_sum = 0.0
    obs_sq_sum = 0.0
    obs_abs_sum = 0.0
    obs_count = 0
    action_counter: Counter[str] = Counter()
    obs_zero_count = 0
    obs_dim: Optional[int] = None

    for episode in range(episodes):
        obs, info = env.reset(seed=seed)
        if reset_hook is not None:
            reset_hook(env)
        obs_sum, obs_sq_sum, obs_abs_sum, obs_count, obs_zero_count = _obs_stats_update(
            obs,
            obs_sum,
            obs_sq_sum,
            obs_abs_sum,
            obs_count,
            obs_zero_count,
        )
        obs_dim = obs_dim or _infer_obs_dim(obs)

        done = False
        ep_return = 0.0
        ep_steps = 0
        crashed = False
        timeout = False

        while not done:
            action = policy(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += float(reward)
            ep_steps += 1
            steps_collected += 1

            crashed = crashed or bool(info.get("crashed", False))
            timeout = timeout or bool(truncated)
            speed = _extract_speed(obs, info)
            if speed is not None:
                speed_samples.append(speed)
            obs_sum, obs_sq_sum, obs_abs_sum, obs_count, obs_zero_count = _obs_stats_update(
                obs,
                obs_sum,
                obs_sq_sum,
                obs_abs_sum,
                obs_count,
                obs_zero_count,
            )
            obs_dim = obs_dim or _infer_obs_dim(obs)

            if track_actions:
                logged_action = action
                if action_getter is not None:
                    try:
                        logged_action = action_getter(action, env)
                    except Exception:
                        logged_action = action
                action_counter[_action_key(logged_action)] += 1

            if max_steps is not None and ep_steps >= max_steps:
                timeout = True
                break

            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(ep_steps)
        collisions += int(crashed)
        timeouts += int(timeout)

    action_distribution = {}
    action_entropy = 0.0
    if track_actions:
        action_total = sum(action_counter.values()) or 1
        action_distribution = {
            key: count / action_total for key, count in action_counter.items()
        }
        action_entropy = -sum(
            p * math.log(max(p, 1e-12)) for p in action_distribution.values()
        )

    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if returns else 0.0
    mean_length = float(np.mean(lengths)) if lengths else 0.0
    std_length = float(np.std(lengths)) if lengths else 0.0
    mean_speed = float(np.mean(speed_samples)) if speed_samples else 0.0
    std_speed = float(np.std(speed_samples)) if speed_samples else 0.0

    obs_mean = obs_sum / obs_count if obs_count else 0.0
    obs_var = obs_sq_sum / obs_count - obs_mean ** 2 if obs_count else 0.0
    obs_var = max(obs_var, 0.0)
    obs_std = float(math.sqrt(obs_var))
    obs_mean_abs = obs_abs_sum / obs_count if obs_count else 0.0
    obs_zero_fraction = (
        obs_zero_count / obs_count if obs_count else 0.0
    )

    return {
        "label": label,
        "episodes": episodes,
        "total_steps": steps_collected,
        "mean_episode_return": mean_return,
        "std_episode_return": std_return,
        "mean_episode_length": mean_length,
        "std_episode_length": std_length,
        "collision_rate": collisions / episodes if episodes else 0.0,
        "timeout_rate": timeouts / episodes if episodes else 0.0,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "action_distribution": action_distribution,
        "action_entropy": action_entropy,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "obs_mean_abs": obs_mean_abs,
        "obs_zero_fraction": obs_zero_fraction,
        "obs_flat_dim": obs_dim or 0,
        "return_min": float(np.min(returns)) if returns else 0.0,
        "return_max": float(np.max(returns)) if returns else 0.0,
        "return_median": float(np.median(returns)) if returns else 0.0,
        "return_p10": float(np.percentile(returns, 10)) if returns else 0.0,
        "return_p90": float(np.percentile(returns, 90)) if returns else 0.0,
        "length_min": float(np.min(lengths)) if lengths else 0.0,
        "length_max": float(np.max(lengths)) if lengths else 0.0,
        "speed_min": float(np.min(speed_samples)) if speed_samples else 0.0,
        "speed_max": float(np.max(speed_samples)) if speed_samples else 0.0,
    }

def _action_key(action: Any) -> str:
    if isinstance(action, dict):
        print(action)
        normalized_items = sorted(
            (str(key), _normalize_action_value(value)) for key, value in action.items()
        )
        return str(normalized_items)
    if isinstance(action, np.ndarray):
        return str(tuple(np.asarray(action).flatten().round(4).tolist()))
    if isinstance(action, (list, tuple)):
        return str(tuple(action))
    if isinstance(action, (np.integer, int)):  # type: ignore[arg-type]
        return str(int(action))
    if isinstance(action, (np.floating, float)):  # type: ignore[arg-type]
        return str(float(action))
    return repr(action)


def _normalize_action_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(np.asarray(value).flatten().round(4).tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_action_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _normalize_action_value(val)) for key, val in value.items())
        )
    if isinstance(value, (np.integer, int)):  # type: ignore[arg-type]
        return int(value)
    if isinstance(value, (np.floating, float)):  # type: ignore[arg-type]
        return float(value)
    return value

def _infer_obs_dim(obs: Any) -> int:
    return int(_flatten_obs(obs).size)


def _obs_stats_update(
    obs: Any,
    obs_sum: float,
    obs_sq_sum: float,
    obs_abs_sum: float,
    obs_count: int,
    obs_zero_count: int,
) -> Tuple[float, float, float, int, int]:
    flat = _flatten_obs(obs)
    obs_sum += float(np.sum(flat))
    obs_sq_sum += float(np.sum(np.square(flat)))
    obs_abs_sum += float(np.sum(np.abs(flat)))
    obs_count += flat.size
    obs_zero_count += int(np.count_nonzero(flat == 0))
    return obs_sum, obs_sq_sum, obs_abs_sum, obs_count, obs_zero_count


def _extract_speed(obs: Any, info: Dict[str, Any]) -> Optional[float]:
    if "speed" in info and isinstance(info["speed"], (int, float)):
        return float(info["speed"])

    try:
        arr = _ensure_vehicle_array(obs)
    except Exception:
        return None

    if arr.size == 0:
        return None

    ego = arr[0]
    if ego.size < 4:
        return None
    vx = float(ego[3])
    vy = float(ego[4]) if ego.size > 4 else 0.0
    return float(math.sqrt(vx * vx + vy * vy))


def _ensure_vehicle_array(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        if "observation" in obs:
            obs = obs["observation"]
        elif "vehicles" in obs:
            obs = obs["vehicles"]
        else:
            obs = next(iter(obs.values()))
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _infer_horizon(env: gym.Env) -> Optional[int]:
    config = getattr(env, "config", None)
    if isinstance(config, dict):
        duration = config.get("duration")
        if isinstance(duration, (int, float)):
            return int(duration)
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _as_float(value: Any) -> Optional[float]:
    if _is_number(value):
        return float(value)
    return None


def _stable_string_id(value: str) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _collect_env_features(env: gym.Env) -> Dict[str, float]:
    base_env = _base_env(env)
    config = getattr(base_env, "config", {}) or {}
    features: Dict[str, float] = {}

    def add(name: str, value: Any) -> None:
        numeric = _as_float(value)
        if numeric is not None:
            features[name] = numeric

    lanes = config.get("lanes_count")
    vehicles_count = config.get("vehicles_count")
    vehicles_density = config.get("vehicles_density")
    duration = config.get("duration")

    add("lanes_count", lanes)
    add("vehicles_count", vehicles_count)
    add("vehicles_density", vehicles_density)
    add("episode_duration", duration)
    add("ego_spacing", config.get("ego_spacing"))
    add("policy_frequency", config.get("policy_frequency"))
    add("simulation_frequency", config.get("simulation_frequency"))
    add("collision_reward", config.get("collision_reward"))
    add("right_lane_reward", config.get("right_lane_reward"))
    add("high_speed_reward", config.get("high_speed_reward"))
    add("lane_change_reward", config.get("lane_change_reward"))
    add("normalize_reward", 1.0 if config.get("normalize_reward") else 0.0)
    reward_speed_range = config.get("reward_speed_range")
    if isinstance(reward_speed_range, (list, tuple)) and len(reward_speed_range) == 2:
        add("reward_speed_min", reward_speed_range[0])
        add("reward_speed_max", reward_speed_range[1])

    target_speed = config.get("target_speed") or config.get("desired_speed")
    add("target_speed_hint", target_speed)
    add("speed_limit_hint", config.get("speed_limit"))

    obs_config = config.get("observation", {})
    if isinstance(obs_config, dict):
        obs_type = obs_config.get("type")
        if isinstance(obs_type, str):
            features["observation_type_id"] = float(_stable_string_id(obs_type))
        add("observation_vehicles_count", obs_config.get("vehicles_count"))
        obs_features = obs_config.get("features")
        if isinstance(obs_features, (list, tuple)):
            add("observation_feature_dim", len(obs_features))

    lanes_val = _as_float(lanes) or 1.0
    veh_cnt_val = _as_float(vehicles_count) or 0.0
    veh_density_val = _as_float(vehicles_density) or 0.0
    duration_val = _as_float(duration) or 1.0
    add(
        "traffic_intensity_index",
        (veh_cnt_val * max(veh_density_val, 1e-3)) / max(lanes_val, 1.0),
    )
    add("vehicles_per_second", veh_cnt_val / max(duration_val, 1.0))

    action_space = env.action_space
    action_type = type(action_space).__name__
    features["action_space_type_id"] = float(_stable_string_id(action_type))
    features["action_is_discrete"] = (
        1.0 if isinstance(action_space, gym.spaces.Discrete) else 0.0
    )
    if isinstance(action_space, gym.spaces.Discrete):
        add("action_space_size", action_space.n)
        add("action_branching_factor", action_space.n)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        add("action_space_size", float(np.prod(action_space.nvec)))
        add("action_branching_factor", float(np.max(action_space.nvec)))
    elif isinstance(action_space, gym.spaces.Box):
        add("action_space_size", int(np.prod(action_space.shape)))
        finite_high = action_space.high[np.isfinite(action_space.high)]
        finite_low = action_space.low[np.isfinite(action_space.low)]
        if finite_high.size:
            add("action_box_high", float(np.max(finite_high)))
        if finite_low.size:
            add("action_box_low", float(np.min(finite_low)))

    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box):
        add("observation_space_dim", int(np.prod(obs_space.shape)))

    return features


def _probe_to_features(probe: Dict[str, Any], prefix: str) -> Dict[str, float]:
    features: Dict[str, float] = {}

    def add(name: str, value: Any) -> None:
        numeric = _as_float(value)
        if numeric is not None:
            features[f"{prefix}_{name}"] = numeric

    for key, value in probe.items():
        if key in {"label", "action_distribution"}:
            continue
        add(key, value)

    mean_return = probe.get("mean_episode_return", 0.0)
    std_return = probe.get("std_episode_return", 0.0)
    mean_length = probe.get("mean_episode_length", 0.0)
    mean_speed = probe.get("mean_speed", 0.0)
    std_speed = probe.get("std_speed", 0.0)
    collision_rate = probe.get("collision_rate", 0.0)
    timeout_rate = probe.get("timeout_rate", 0.0)

    add("return_per_step", mean_return / max(mean_length, 1.0))
    add("reward_snr", mean_return / (std_return + 1e-6))
    add("speed_cv", std_speed / (abs(mean_speed) + 1e-6))
    add("safety_index", 1.0 - collision_rate)
    add("timeout_pressure", timeout_rate)
    add("safety_vs_speed", (1.0 - collision_rate) * mean_speed)
    return features


def _derived_features(
    random_probe: Dict[str, Any],
    idm_probe: Dict[str, Any],
    env_features: Dict[str, float],
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    r_return = random_probe.get("mean_episode_return", 0.0)
    i_return = idm_probe.get("mean_episode_return", 0.0)
    r_std = random_probe.get("std_episode_return", 0.0)
    i_std = idm_probe.get("std_episode_return", 0.0)
    r_speed = random_probe.get("mean_speed", 0.0)
    i_speed = idm_probe.get("mean_speed", 0.0)
    r_collision = random_probe.get("collision_rate", 0.0)
    i_collision = idm_probe.get("collision_rate", 0.0)
    r_timeout = random_probe.get("timeout_rate", 0.0)
    i_timeout = idm_probe.get("timeout_rate", 0.0)
    r_obs_dim = random_probe.get("obs_flat_dim", 0.0)
    entropy = random_probe.get("action_entropy", 0.0)

    features["idm_return_gain"] = i_return - r_return
    features["idm_return_ratio"] = i_return / (abs(r_return) + 1e-6)
    features["idm_collision_reduction"] = max(r_collision - i_collision, 0.0)
    features["idm_timeout_reduction"] = r_timeout - i_timeout
    features["idm_speed_gain"] = i_speed - r_speed
    features["random_reward_stochasticity"] = r_std / (abs(r_return) + 1e-6)
    features["stability_gap"] = r_std - i_std
    features["action_entropy_random"] = entropy
    features["obs_complexity"] = r_obs_dim
    features["difficulty_index"] = (
        env_features.get("traffic_intensity_index", 0.0)
        * features["random_reward_stochasticity"]
    )
    features["safety_gap"] = (1.0 - i_collision) - (1.0 - r_collision)
    features["speed_safety_tradeoff"] = (
        i_speed * (1.0 - i_collision) - r_speed * (1.0 - r_collision)
    )
    features["timeout_sensitivity"] = r_timeout
    return features
