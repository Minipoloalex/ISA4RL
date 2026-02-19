import hashlib
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import gymnasium as gym
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

from configs import InstanceConfig
from utils import _flatten_obs

PolicyFn = Callable[[Any, Dict[str, Any]], Any]

INITIAL_GEOMETRY_SAMPLES = 3
MAX_FREE_SPACE_METERS = 200.0
MAX_TTC_SECONDS = 30.0


@dataclass
class StepMetricsContext:
    env: gym.Env
    pre_obs: Any
    post_obs: Any
    action: Any
    reward: float
    info: Dict[str, Any]
    prev_info: Optional[Dict[str, Any]]
    prev_action: Any
    prev_reward: Optional[float]
    episode_index: int
    step_index: int
    terminated: bool
    truncated: bool


class MetricHook(Protocol):
    def on_probe_start(self) -> None:
        ...

    def on_episode_start(self) -> None:
        ...

    def on_step(self, context: StepMetricsContext) -> None:
        ...

    def on_episode_end(self) -> None:
        ...

    def finalize(self) -> Dict[str, float]:
        ...


def extract_metafeatures(config: InstanceConfig):
    before = time.perf_counter()
    env = config.ensure_eval_env()
    env_name = env.spec.id
    env_seed = config.eval_seed
    env_features = _collect_env_features(env)
    # env_features.update(_initial_geometry_features(env, env_seed))

    random_probe = _run_probe(
        env=env,
        policy=_make_random_policy(env),
        episodes=10,
        env_seed=env_seed,
        label="random_rollout",
        metric_hooks=_build_metric_hooks(),
    )
    STUFF_IDK = [
        # IDK what these are and if they may be decent
        # Or just IDK if I want to keep them
        "conflict_rate",
        "lane_change_feasible_rate_left",
        "lane_change_feasible_rate_right",
        "nearby_vehicles_10m",
        "nearby_vehicles_20m",
        "nearby_vehicles_30m",
    ]
    INCLUDE_STUFF = [
        "total_steps",
        "mean_episode_return",
        "std_episode_return",
        "mean_episode_length",
        "std_episode_length",
        "collision_rate",
        "timeout_rate",
        "mean_speed",
        "std_speed",
        "reward_sparsity",
        "reward_skew",
        "reward_kurtosis",
        "reward_autocorr1",
        "corr_reward_speed",    # are these actually good?
        "corr_reward_lane_change",
        "corr_reward_progress",
    ]
    idm_probe = _run_idm_probe(
        env=env,
        episodes=5,
        env_seed=env_seed,
        label="idm_like",
    )
    config.close()
    elapsed = time.perf_counter() - before

    idm_advantage = (
        idm_probe["mean_episode_return"] - random_probe["mean_episode_return"]
    )
    safety_delta = random_probe["collision_rate"] - idm_probe["collision_rate"]
    features: Dict[str, float] = {}
    features.update(env_features)
    features.update(_random_probe_features(random_probe))
    features.update(_baseline_probe_features(idm_probe, "idm"))
    features.update(
        _derived_features(
            random_probe=random_probe,
            idm_probe=idm_probe,
            env_features=env_features,
        )
    )
    return {
        "instance_id": config.id,
        "env_config_id": config.id_env_config,
        "obs_config_id": config.id_obs_config,
        "env_name": env_name,
        "generated_at": time.time_ns(),
        "elapsed_seconds": elapsed,
        "probes": {
            "random": random_probe,
            "idm_like": idm_probe,
        },
        "diagnostics": {
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
    env_seed: int,
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
        env_seed=env_seed,
        label=label,
        reset_hook=reset_hook,
        track_actions=True,
        action_getter=_idm_action_getter,
        metric_hooks=_build_metric_hooks(),
    )


def _default_idle_action(env: gym.Env) -> int:
    if isinstance(env.action_space, gym.spaces.Discrete):
        return 0
    raise NotImplementedError(f"Idle action not implemented for action space type: {type(env.action_space)}")

def _constant_policy(action: Any) -> PolicyFn:
    def policy(_: Any, __: Dict[str, Any]) -> Any:
        return action

    return policy


def _reset_policy_state(policy: Any) -> None:
    # TODO: check if this function does anything
    if hasattr(policy, "reset_episode"):
        try:
            policy.reset_episode()
        except Exception:
            pass
    elif hasattr(policy, "reset"):
        try:
            policy.reset()
        except Exception:
            pass


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


def _ensure_idm_vehicle(env: gym.Env) -> None:
    base_env = env.unwrapped
    vehicle = base_env.vehicle  # type: ignore
    road = base_env.road    # type: ignore
    # vehicle = getattr(base_env, "vehicle", None)
    # road = getattr(base_env, "road", None)

    idm_vehicle = IDMVehicle.create_from(vehicle)
    idm_vehicle.enable_lane_change = True
    idm_vehicle.timer = getattr(vehicle, "timer", getattr(idm_vehicle, "timer", 0.0))
    _copy_vehicle_attr(vehicle, idm_vehicle, "target_speed")
    _copy_vehicle_attr(vehicle, idm_vehicle, "target_speeds", deep_copy=True)
    _copy_vehicle_attr(vehicle, idm_vehicle, "index_to_speed")
    _copy_vehicle_attr(vehicle, idm_vehicle, "speed_index")

    base_env.vehicle = idm_vehicle  # type: ignore

    for idx, existing in enumerate(road.vehicles):
        if existing is vehicle:
            road.vehicles[idx] = idm_vehicle
            break

    base_env.action_type.controlled_vehicle = idm_vehicle   # type: ignore
    # action_type = getattr(base_env, "action_type", None)
    # if action_type is not None:
    #     action_type.controlled_vehicle = idm_vehicle


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
    vehicle = env.unwrapped.vehicle # type: ignore
    action = vehicle.action
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
    env_seed: int,
    label: str,
    reset_hook: Optional[Callable[[gym.Env], None]] = None,
    track_actions: bool = True,
    action_getter: Optional[Callable[[Any, gym.Env], Any]] = None,
    metric_hooks: Optional[List[MetricHook]] = None,
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

    metric_hooks = metric_hooks or []
    for hook in metric_hooks:
        try:
            hook.on_probe_start()
        except Exception:
            continue

    for episode in range(episodes):
        obs, info = env.reset(seed=env_seed)
        if reset_hook is not None:
            reset_hook(env)
        _reset_policy_state(policy)
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
        prev_action: Any = None
        prev_reward: Optional[float] = None

        for hook in metric_hooks:
            try:
                hook.on_episode_start()
            except Exception:
                continue

        while not done:
            pre_step_obs = obs
            pre_step_info = info
            action = policy(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += float(reward)
            ep_steps += 1
            steps_collected += 1

            crashed = crashed or bool(info.get("crashed", False))
            timeout = timeout or bool(truncated)
            speed = info.get("speed")
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

            if metric_hooks:
                step_context = StepMetricsContext(
                    env=env,
                    pre_obs=pre_step_obs,
                    post_obs=obs,
                    action=action,
                    reward=float(reward),
                    info=info or {},
                    prev_info=pre_step_info,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                    episode_index=episode,
                    step_index=ep_steps,
                    terminated=terminated,
                    truncated=truncated,
                )
                for hook in metric_hooks:
                    try:
                        hook.on_step(step_context)
                    except Exception:
                        continue

            if track_actions:
                logged_action = action
                if action_getter is not None:
                    try:
                        logged_action = action_getter(action, env)
                    except Exception:
                        logged_action = action
                action_counter[_action_key(logged_action)] += 1

            done = terminated or truncated
            prev_action = action
            prev_reward = float(reward)

        returns.append(ep_return)
        lengths.append(ep_steps)
        collisions += int(crashed)
        timeouts += int(timeout)

        for hook in metric_hooks:
            try:
                hook.on_episode_end()
            except Exception:
                continue

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))
    mean_speed = float(np.mean(speed_samples))
    std_speed = float(np.std(speed_samples))

    obs_mean = obs_sum / obs_count
    obs_var = obs_sq_sum / obs_count - obs_mean ** 2
    obs_var = max(obs_var, 0.0)
    obs_std = float(math.sqrt(obs_var))
    obs_mean_abs = obs_abs_sum / obs_count
    obs_zero_fraction = (
        obs_zero_count / obs_count
    )

    result = {
        "label": label,
        "episodes": episodes,
        "total_steps": steps_collected,
        "mean_episode_return": mean_return,
        "std_episode_return": std_return,
        "mean_episode_length": mean_length,
        "std_episode_length": std_length,
        "collision_rate": collisions / episodes,
        "timeout_rate": timeouts / episodes,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        # "obs_mean": obs_mean,
        # "obs_std": obs_std,
        "obs_mean_abs": obs_mean_abs,
        "obs_zero_fraction": obs_zero_fraction,
        "obs_flat_dim": obs_dim or 0,
        "collision_observed": 1.0 if collisions > 0 else 0.0,
        "timeout_observed": 1.0 if timeouts > 0 else 0.0,
        # "return_min": float(np.min(returns)),
        # "return_max": float(np.max(returns)),
        # "return_median": float(np.median(returns)),
        # "length_min": float(np.min(lengths)),
        # "length_max": float(np.max(lengths)),
        # "speed_min": float(np.min(speed_samples)),
        # "speed_max": float(np.max(speed_samples)),
    }

    extra_metrics: Dict[str, float] = {}
    for hook in metric_hooks:
        try:
            extra_metrics.update(hook.finalize())
        except Exception:
            continue

    result.update(extra_metrics)
    return result

def _action_key(action: Any) -> str:
    if isinstance(action, dict):
        # print(action)
        normalized_items = sorted(
            (str(key), float(value)) for key, value in action.items()
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


def _copy_action_like(action: Any) -> Any:
    if isinstance(action, dict):
        return {key: _copy_action_like(value) for key, value in action.items()}
    if isinstance(action, np.ndarray):
        return np.array(action, copy=True)
    if isinstance(action, list):
        return [_copy_action_like(value) for value in action]
    if isinstance(action, tuple):
        return tuple(_copy_action_like(value) for value in action)
    return action


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

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))

def _as_float(value: Any) -> Optional[float]:
    if _is_number(value):
        return float(value)
    return None

def _collect_env_features(env: gym.Env) -> Dict[str, float]:
    base_env = env.unwrapped
    config = getattr(base_env, "config", {}) or {}
    features: Dict[str, float] = {}

    def add(name: str, value: Any, default: Optional[Any] = None) -> None:
        numeric = _as_float(value)
        if numeric is not None:
            features[name] = numeric
        elif default is not None:
            features[name] = default

    # roundabout and merge don't have these parameters (their values are hardcoded to these)
    lanes = config.get("lanes_count", 2)
    vehicles_count = config.get("vehicles_count",4)
    vehicles_density = config.get("vehicles_density", 1)

    collision_reward = config.get("collision_reward")
    right_lane_reward = config.get("right_lane_reward")
    high_speed_reward = config.get("high_speed_reward")
    lane_change_reward = config.get("lane_change_reward")

    add("lanes_count", lanes)
    add("vehicles_count", vehicles_count)
    add("vehicles_density", vehicles_density)
    add("collision_reward", collision_reward)
    add("right_lane_reward", right_lane_reward)
    add("high_speed_reward", high_speed_reward)
    add("lane_change_reward", lane_change_reward)

    action_space = env.action_space
    add("action_space_size", action_space.n)

    obs_space = env.observation_space
    assert(isinstance(obs_space, gym.spaces.Box))
    add("observation_space_dim", int(np.prod(obs_space.shape)))

    return features


def _initial_geometry_features(
    env: gym.Env,
    env_seed: int,
    samples: int = INITIAL_GEOMETRY_SAMPLES,
) -> Dict[str, float]:
    snapshots: List[Dict[str, float]] = []
    env.reset(seed=env_seed)
    snapshot = _geometry_snapshot_from_env(env)
    snapshots.append(snapshot)

    free_space_ahead = [_cap_free_space(s["free_space_ahead"]) for s in snapshots]
    free_space_any = [_cap_free_space(s["free_space_min"]) for s in snapshots]
    lane_entropy = [s["lane_entropy"] for s in snapshots]
    relative_speed_std = [s["relative_speed_std"] for s in snapshots]
    nearby_counts = [s["vehicles_within_20m"] for s in snapshots]

    print(free_space_ahead)
    features: Dict[str, float] = {}
    features["init_free_space_ahead_mean"] = float(np.mean(free_space_ahead))
    features["init_free_space_ahead_min"] = float(np.min(free_space_ahead))
    features["init_free_space_min"] = float(np.min(free_space_any))
    features["init_lane_occupancy_entropy"] = float(np.mean(lane_entropy))
    features["init_relative_speed_std"] = float(np.mean(relative_speed_std))
    features["init_num_vehicles_within_20m"] = float(np.mean(nearby_counts))
    if len(free_space_ahead) > 1:
        features["initial_state_stochasticity"] = float(np.std(free_space_ahead))
    else:
        features["initial_state_stochasticity"] = 0.0
    return features


def _geometry_snapshot_from_env(env: gym.Env) -> Optional[Dict[str, float]]:
    base_env = env.unwrapped
    road = getattr(base_env, "road", None)
    ego = getattr(base_env, "vehicle", None)
    if road is None or ego is None:
        return None
    ego_lane = _lane_id_from_vehicle(ego)
    if ego_lane is None:
        return None
    ego_pos = _vehicle_position(ego)
    if ego_pos.size < 2:
        return None

    front_gap = math.inf
    any_front_gap = math.inf
    lane_counts: Counter = Counter()
    rel_speed_samples: List[float] = []
    nearby_within_20 = 0

    vehicles = getattr(road, "vehicles", [])
    for other in vehicles:
        if other is ego:
            continue
        lane_id = _lane_id_from_vehicle(other)
        if lane_id is None:
            continue
        lane_counts[lane_id] += 1
        other_pos = _vehicle_position(other)
        if other_pos.size < 2:
            continue
        gap = float(other_pos[0] - ego_pos[0])
        if gap >= 0.0:
            any_front_gap = min(any_front_gap, gap)
            if lane_id == ego_lane:
                front_gap = min(front_gap, gap)
        distance = float(np.linalg.norm(other_pos - ego_pos))
        if distance <= 20.0:
            nearby_within_20 += 1
        rel_speed_samples.append(_vehicle_speed(other) - ego.speed)

    snapshot: Dict[str, float] = {}
    snapshot["free_space_ahead"] = float(front_gap) if math.isfinite(front_gap) else float("inf")
    snapshot["free_space_min"] = float(any_front_gap) if math.isfinite(any_front_gap) else float("inf")
    snapshot["lane_entropy"] = _entropy_from_counts(lane_counts)
    snapshot["relative_speed_std"] = (
        float(np.std(rel_speed_samples)) if rel_speed_samples else 0.0
    )
    snapshot["vehicles_within_20m"] = float(nearby_within_20)
    return snapshot


def _cap_free_space(value: float) -> float:
    if not math.isfinite(value):
        return MAX_FREE_SPACE_METERS
    return float(min(value, MAX_FREE_SPACE_METERS))


def _entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probabilities = np.asarray(list(counts.values()), dtype=float) / float(total)
    probabilities = probabilities[probabilities > 0]
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))


def _random_probe_features(probe: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    keep_keys = [
        "mean_episode_return",
        "std_episode_return",
        "collision_rate",
        "timeout_rate",
        "mean_speed",
        "reward_sparsity",
        "reward_skew",
        "reward_kurtosis",
        "reward_autocorr1",
        "collision_observed",
        "timeout_observed",
    ]
    for key in keep_keys:
        _add_prefixed_feature(features, "random", key, probe.get(key))

    mean_return = _probe_value(probe, "mean_episode_return")
    std_return = _probe_value(probe, "std_episode_return")
    mean_length = _probe_value(probe, "mean_episode_length")
    mean_speed = _probe_value(probe, "mean_speed")
    std_speed = _probe_value(probe, "std_speed")
    collision_rate = _probe_value(probe, "collision_rate")
    timeout_rate = _probe_value(probe, "timeout_rate")

    if mean_return is not None and mean_length is not None and mean_length > 0.0:
        features["random_return_per_step"] = mean_return / max(mean_length, 1e-6)
    if mean_return is not None and std_return is not None:
        features["random_reward_snr"] = mean_return / (abs(std_return) + 1e-6)
    if mean_speed is not None and std_speed is not None:
        denom = max(abs(mean_speed), 1e-6)
        features["random_speed_cv"] = std_speed / denom
    if collision_rate is not None:
        features["random_safety_index"] = 1.0 - collision_rate
        if mean_speed is not None:
            features["random_safety_vs_speed"] = (1.0 - collision_rate) * mean_speed
    if timeout_rate is not None:
        features["random_timeout_pressure"] = timeout_rate

    for ttc_key in ["ttc_p05", "ttc_p50", "ttc_p95"]:
        _add_prefixed_feature(
            features,
            "random",
            ttc_key,
            probe.get(ttc_key),
            fallback=MAX_TTC_SECONDS,
            impute_flag=f"random_{ttc_key}_imputed",
        )
    _add_prefixed_feature(
        features,
        "random",
        "headway_mean",
        probe.get("headway_mean"),
        fallback=MAX_FREE_SPACE_METERS,
        impute_flag="random_headway_mean_imputed",
    )
    return features


def _baseline_probe_features(probe: Dict[str, Any], prefix: str) -> Dict[str, float]:
    features: Dict[str, float] = {}
    keep_keys = [
        "mean_episode_return",
        "collision_rate",
        "timeout_rate",
        "mean_speed",
        "collision_observed",
        "timeout_observed",
    ]
    for key in keep_keys:
        _add_prefixed_feature(features, prefix, key, probe.get(key))

    mean_speed = _probe_value(probe, "mean_speed")
    std_speed = _probe_value(probe, "std_speed")
    if mean_speed is not None and std_speed is not None:
        denom = max(abs(mean_speed), 1e-6)
        features[f"{prefix}_speed_cv"] = std_speed / denom

    collision_rate = _probe_value(probe, "collision_rate")
    if collision_rate is not None:
        features[f"{prefix}_safety_index"] = 1.0 - collision_rate
    return features


def _add_prefixed_feature(
    features: Dict[str, float],
    prefix: str,
    name: str,
    value: Any,
    *,
    fallback: Optional[float] = None,
    impute_flag: Optional[str] = None,
) -> None:
    key = f"{prefix}_{name}" if prefix else name
    numeric = _as_float(value)
    used_fallback = False
    if numeric is None or not math.isfinite(numeric):
        if fallback is None:
            return
        numeric = fallback
        used_fallback = True
    features[key] = float(numeric)
    if impute_flag is not None:
        features[impute_flag] = 1.0 if used_fallback else 0.0


def _probe_value(probe: Dict[str, Any], key: str) -> Optional[float]:
    value = probe.get(key)
    if value is None:
        return None
    return _as_float(value)


def _probe_stat(probe: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = _probe_value(probe, key)
    if value is None:
        return default
    return value


def _derived_features(
    random_probe: Dict[str, Any],
    idm_probe: Dict[str, Any],
    env_features: Dict[str, float],
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    r_return = _probe_stat(random_probe, "mean_episode_return")
    i_return = _probe_stat(idm_probe, "mean_episode_return")

    r_std = _probe_stat(random_probe, "std_episode_return")
    i_std = _probe_stat(idm_probe, "std_episode_return")
    r_speed = _probe_stat(random_probe, "mean_speed")
    i_speed = _probe_stat(idm_probe, "mean_speed")

    r_collision = _probe_stat(random_probe, "collision_rate")
    i_collision = _probe_stat(idm_probe, "collision_rate")

    r_timeout = _probe_stat(random_probe, "timeout_rate")
    i_timeout = _probe_stat(idm_probe, "timeout_rate")

    r_obs_dim = _probe_stat(random_probe, "obs_flat_dim")

    features["idm_return_gain"] = i_return - r_return
    features["idm_return_ratio"] = i_return / (abs(r_return) + 1e-6)
    features["idm_collision_reduction"] = max(r_collision - i_collision, 0.0)
    features["idm_timeout_reduction"] = r_timeout - i_timeout
    features["idm_speed_gain"] = i_speed - r_speed
    features["random_reward_stochasticity"] = r_std / (abs(r_return) + 1e-6)
    features["stability_gap"] = r_std - i_std
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

    returns = [r_return, i_return]
    collisions = [r_collision, i_collision]
    timeouts = [r_timeout, i_timeout]
    speeds = [r_speed, i_speed]

    features["probe_return_spread"] = max(returns) - min(returns)
    features["probe_safety_spread"] = max(collisions) - min(collisions)
    features["probe_timeout_spread"] = max(timeouts) - min(timeouts)
    features["probe_speed_spread"] = max(speeds) - min(speeds)
    return features

class BaseMetricHook:
    def on_probe_start(self) -> None:
        pass

    def on_episode_start(self) -> None:
        pass

    def on_step(self, context: StepMetricsContext) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def finalize(self) -> Dict[str, float]:
        return {}

class TrafficMetricHook(BaseMetricHook):
    def __init__(
        self,
        lane_gap_threshold: float = 10.0,
        conflict_ttc_threshold: float = 1.5,
    ) -> None:
        self.lane_gap_threshold = lane_gap_threshold
        self.conflict_ttc_threshold = conflict_ttc_threshold
        self._snapshots: List[Dict[str, float]] = []

    def on_step(self, context: StepMetricsContext) -> None:
        snapshot = _traffic_snapshot(context, self.lane_gap_threshold, self.conflict_ttc_threshold)
        if snapshot:
            self._snapshots.append(snapshot)

    def finalize(self) -> Dict[str, float]:
        if not self._snapshots:
            return {}
        result: Dict[str, float] = {}
        min_ttc_values = np.array(
            [snap["min_ttc"] for snap in self._snapshots if math.isfinite(snap["min_ttc"])],
            dtype=float,
        )
        if min_ttc_values.size:
            result["ttc_min"] = float(np.min(min_ttc_values))
            result["ttc_p05"] = float(np.percentile(min_ttc_values, 5))
            result["ttc_p50"] = float(np.percentile(min_ttc_values, 50))
            result["ttc_p95"] = float(np.percentile(min_ttc_values, 95))

        def _mean_from_snapshots(key: str) -> Optional[float]:
            values = [snap[key] for snap in self._snapshots if key in snap]
            if not values:
                return None
            return float(np.mean(values))

        conflict_rate = _mean_from_snapshots("conflict_flag")
        if conflict_rate is not None:
            result["conflict_rate"] = conflict_rate

        feasible_left = _mean_from_snapshots("lane_change_feasible_left")
        if feasible_left is not None:
            result["lane_change_feasible_rate_left"] = feasible_left

        feasible_right = _mean_from_snapshots("lane_change_feasible_right")
        if feasible_right is not None:
            result["lane_change_feasible_rate_right"] = feasible_right

        headways = [snap["headway_m"] for snap in self._snapshots if math.isfinite(snap["headway_m"])]
        if headways:
            result["headway_mean"] = float(np.mean(headways))

        closing = [snap["closing_speed_mps"] for snap in self._snapshots if snap["closing_speed_mps"] is not None]
        if closing:
            result["closing_speed_mean"] = float(np.mean(closing))

        for radius in (10, 20, 30):
            counts = [snap[f"nearby_count_{radius}m"] for snap in self._snapshots if f"nearby_count_{radius}m" in snap]
            if counts:
                result[f"nearby_vehicles_{radius}m"] = float(np.mean(counts))

        for lane_key in [
            "ttc_front",
            "ttc_rear",
            "ttc_left_front",
            "ttc_left_rear",
            "ttc_right_front",
            "ttc_right_rear",
        ]:
            values = [
                snap[lane_key]
                for snap in self._snapshots
                if lane_key in snap and math.isfinite(snap[lane_key])
            ]
            if values:
                result[lane_key] = float(np.mean(values))
        return result


class RewardSignalHook(BaseMetricHook):
    def __init__(self, sparsity_epsilon: float = 1e-6) -> None:
        self.sparsity_epsilon = sparsity_epsilon
        self._rewards: List[float] = []
        self._speed_values: List[float] = []
        self._speed_rewards: List[float] = []
        self._lane_change_flags: List[float] = []
        self._lane_change_rewards: List[float] = []
        self._progress_deltas: List[float] = []
        self._progress_rewards: List[float] = []
        self._episode_sequences: List[List[float]] = []
        self._episode_buffer: List[float] = []
        self._prev_lane: Optional[int] = None
        self._prev_progress: Optional[float] = None

    def on_episode_start(self) -> None:
        self._episode_buffer = []
        self._prev_lane = None
        self._prev_progress = None

    def on_step(self, context: StepMetricsContext) -> None:
        reward = float(context.reward)
        self._rewards.append(reward)
        self._episode_buffer.append(reward)

        speed = context.info.get("speed")
        if speed is not None:
            self._speed_values.append(float(speed))
            self._speed_rewards.append(reward)

        base_env = context.env.unwrapped
        vehicle = getattr(base_env, "vehicle", None)
        lane_id = _lane_id_from_vehicle(vehicle) if vehicle is not None else None
        if lane_id is not None:
            if self._prev_lane is None:
                lane_change_flag = 0.0
            else:
                lane_change_flag = 1.0 if lane_id != self._prev_lane else 0.0
            self._lane_change_flags.append(lane_change_flag)
            self._lane_change_rewards.append(reward)
            self._prev_lane = lane_id

        progress = None
        if vehicle is not None:
            pos = _vehicle_position(vehicle)
            if pos.size:
                progress = float(pos[0])
        if progress is not None:
            if self._prev_progress is not None:
                delta = progress - self._prev_progress
                self._progress_deltas.append(delta)
                self._progress_rewards.append(reward)
            self._prev_progress = progress

    def on_episode_end(self) -> None:
        if self._episode_buffer:
            self._episode_sequences.append(list(self._episode_buffer))
        self._episode_buffer = []

    def finalize(self) -> Dict[str, float]:
        if not self._rewards:
            return {}

        rewards = np.asarray(self._rewards, dtype=float)
        stats: Dict[str, float] = {}
        abs_rewards = np.abs(rewards)
        stats["reward_sparsity"] = float(np.mean(abs_rewards < self.sparsity_epsilon))

        mean_reward = float(np.mean(rewards))
        centered = rewards - mean_reward
        std_reward = float(np.std(rewards))
        if std_reward > 1e-8:
            stats["reward_skew"] = float(np.mean(centered ** 3) / (std_reward ** 3))
            stats["reward_kurtosis"] = float(np.mean(centered ** 4) / (std_reward ** 4))
        else:
            stats["reward_skew"] = 0.0
            stats["reward_kurtosis"] = 0.0

        autocorr = _safe_autocorr(rewards)
        if autocorr is not None:
            stats["reward_autocorr1"] = autocorr

        corr_speed = _safe_corr(self._speed_rewards, self._speed_values)
        if corr_speed is not None:
            stats["corr_reward_speed"] = corr_speed

        corr_lane = _safe_corr(self._lane_change_rewards, self._lane_change_flags)
        if corr_lane is not None:
            stats["corr_reward_lane_change"] = corr_lane

        corr_progress = _safe_corr(self._progress_rewards, self._progress_deltas)
        if corr_progress is not None:
            stats["corr_reward_progress"] = corr_progress

        return stats



def _build_metric_hooks() -> List[MetricHook]:
    return [
        # TrafficMetricHook(),
        RewardSignalHook(),
    ]


def _traffic_snapshot(
    context: StepMetricsContext,
    lane_gap_threshold: float,
    conflict_ttc_threshold: float,
) -> Optional[Dict[str, float]]:
    base_env = context.env.unwrapped
    road = getattr(base_env, "road", None)
    ego = getattr(base_env, "vehicle", None)
    if road is None or ego is None:
        return None

    vehicles = getattr(road, "vehicles", [])
    ego_lane = _lane_id_from_vehicle(ego)
    ego_speed = ego.speed
    ego_pos = _vehicle_position(ego)
    if ego_pos.size < 2:
        return None
    if ego_lane is None:
        return None

    lane_stats: Dict[str, Dict[str, Any]] = {
        "current": {"front_gap": math.inf, "rear_gap": math.inf, "front_speed": None, "rear_speed": None},
        "left": {"front_gap": math.inf, "rear_gap": math.inf, "front_speed": None, "rear_speed": None},
        "right": {"front_gap": math.inf, "rear_gap": math.inf, "front_speed": None, "rear_speed": None},
    }
    nearby_counts = {10: 0, 20: 0, 30: 0}

    for other in vehicles:
        if other is ego:
            continue
        lane_id = _lane_id_from_vehicle(other)
        if lane_id is None or ego_lane is None:
            continue
        lane_delta = lane_id - ego_lane
        lane_key = None
        if lane_delta == 0:
            lane_key = "current"
        elif lane_delta == 1:
            lane_key = "left"
        elif lane_delta == -1:
            lane_key = "right"
        if lane_key is None:
            continue

        other_pos = _vehicle_position(other)
        if other_pos.size < 2:
            continue
        longitudinal_gap = float(other_pos[0] - ego_pos[0])
        abs_gap = abs(longitudinal_gap)
        for radius in nearby_counts:
            if abs_gap <= radius:
                nearby_counts[radius] += 1

        lane_entry = lane_stats[lane_key]
        other_speed = _vehicle_speed(other)
        if longitudinal_gap >= 0.0:
            if longitudinal_gap < lane_entry["front_gap"]:
                lane_entry["front_gap"] = longitudinal_gap
                lane_entry["front_speed"] = other_speed
        else:
            rear_gap = abs(longitudinal_gap)
            if rear_gap < lane_entry["rear_gap"]:
                lane_entry["rear_gap"] = rear_gap
                lane_entry["rear_speed"] = other_speed

    ttc_samples = []
    snapshot: Dict[str, float] = {}
    current_lane = lane_stats["current"]
    headway = current_lane["front_gap"]
    snapshot["headway_m"] = float(headway)
    closing_speed = None
    if math.isfinite(headway) and current_lane["front_speed"] is not None:
        closing_speed = max(ego_speed - current_lane["front_speed"], 0.0)
    snapshot["closing_speed_mps"] = closing_speed

    ttc_front = _ttc_from_gap(headway, closing_speed)
    snapshot["ttc_front"] = ttc_front
    ttc_samples.append(ttc_front)

    ttc_rear = _rear_ttc(current_lane["rear_gap"], current_lane["rear_speed"], ego_speed)
    snapshot["ttc_rear"] = ttc_rear
    ttc_samples.append(ttc_rear)

    for side, prefix in (("left", "ttc_left"), ("right", "ttc_right")):
        lane_entry = lane_stats[side]
        front_ttc = _ttc_from_gap(lane_entry["front_gap"], _closing_speed(ego_speed, lane_entry["front_speed"]))
        rear_ttc = _rear_ttc(lane_entry["rear_gap"], lane_entry["rear_speed"], ego_speed)
        snapshot[f"{prefix}_front"] = front_ttc
        snapshot[f"{prefix}_rear"] = rear_ttc
        ttc_samples.extend([front_ttc, rear_ttc])
        snapshot[f"lane_change_feasible_{side}"] = float(
            (lane_entry["front_gap"] > lane_gap_threshold)
            and (lane_entry["rear_gap"] > lane_gap_threshold)
        )

    finite_ttc = [value for value in ttc_samples if math.isfinite(value)]
    snapshot["min_ttc"] = float(min(finite_ttc)) if finite_ttc else float("inf")
    snapshot["conflict_flag"] = 1.0 if snapshot["min_ttc"] < conflict_ttc_threshold else 0.0

    snapshot["lane_change_feasible_left"] = snapshot.get("lane_change_feasible_left", 0.0)
    snapshot["lane_change_feasible_right"] = snapshot.get("lane_change_feasible_right", 0.0)
    
    for radius, count in nearby_counts.items():
        snapshot[f"nearby_count_{radius}m"] = float(count)
    return snapshot


def _ttc_from_gap(gap: float, closing_speed: Optional[float]) -> float:
    if not math.isfinite(gap) or closing_speed is None or closing_speed <= 1e-6:
        return float("inf")
    return float(max(gap, 0.0) / max(closing_speed, 1e-6))


def _closing_speed(ego_speed: float, other_speed: Optional[float]) -> Optional[float]:
    if other_speed is None:
        return None
    return max(ego_speed - other_speed, 0.0)


def _rear_ttc(gap: float, other_speed: Optional[float], ego_speed: float) -> float:
    if not math.isfinite(gap) or other_speed is None:
        return float("inf")
    rel_speed = max(other_speed - ego_speed, 0.0)
    if rel_speed <= 1e-6:
        return float("inf")
    return float(gap / rel_speed)


def _vehicle_position(vehicle: Any) -> np.ndarray:
    position = getattr(vehicle, "position", None)
    if position is None:
        return np.zeros(2, dtype=float)
    return np.asarray(position, dtype=float)


def _lane_id_from_vehicle(vehicle: Any) -> Optional[int]:
    if vehicle is None:
        return None
    lane_index = getattr(vehicle, "lane_index", None)
    if isinstance(lane_index, (list, tuple)) and lane_index:
        lane_index = lane_index[-1]
    if isinstance(lane_index, (np.integer, int)):
        return int(lane_index)
    return None


def _vehicle_speed(vehicle: Any) -> float:
    return vehicle.speed

def _safe_autocorr(values: np.ndarray) -> Optional[float]:
    if values.size <= 1:
        return None
    x = values[:-1]
    y = values[1:]
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_corr(x_values: List[float], y_values: List[float]) -> Optional[float]:
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return None
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

