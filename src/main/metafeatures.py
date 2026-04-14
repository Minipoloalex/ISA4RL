import hashlib
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, SupportsFloat
import gymnasium as gym
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from utils.metafeatures.metrics import SimpleEgoMetricsHook
from utils.general_utils import as_float

from configs import InstanceConfig
from utils.general_utils import _flatten_obs
from utils.metafeature_utils import (
    Trajectory,
    StepInfo,
    make_random_policy,
    constant_policy,
    default_idle_action,
    ensure_idm_vehicle,
    PolicyFn,
)

INITIAL_GEOMETRY_SAMPLES = 3
MAX_FREE_SPACE_METERS = 200.0
MAX_TTC_SECONDS = 30.0


def extract_metafeatures(config: InstanceConfig):
    before = time.perf_counter()
    env = config.ensure_test_env()
    env_name = env.spec.id
    env_features = _collect_env_features(env_name, env)

    trajectories_random: List[Trajectory] = _run_probe(
        env=env,
        policy=make_random_policy(env),
        episodes=config.n_test_episodes,
    )
    trajectories_idm = _run_idm_probe(
        env=env,
        episodes=config.n_test_episodes,
    )

    # save_trajectories("random", trajectories_random, config)
    # save_trajectories("idm", trajectories_idm, config)
    config.close()
    metafeatures_random = traj_metafeatures("random", trajectories_random)
    metafeatures_idm = traj_metafeatures("idm", trajectories_idm)
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


def traj_metafeatures(label: str, trajectories: List[Trajectory]) -> Dict[str, Any]:
    # Initialize the hook we built earlier
    hooks = [SimpleEgoMetricsHook()]
    for hook in hooks:
        hook.on_probe_start()

    for traj in trajectories:
        for hook in hooks:
            hook.on_episode_start()
        
        for step in traj:
            for hook in hooks:
                hook.on_step(step)
            
        for hook in hooks:
            hook.on_episode_end()

    # Retrieve the aggregated stats
    for hook in hooks:
        metrics = hook.finalize()

    # Format the prefix to ensure clean dictionary keys (e.g., 'eval/reward_mean')
    prefix = f"{label}/"

    # Return the new dictionary with prefixed keys
    return {f"{prefix}{key}": value for key, value in metrics.items()}

def _run_idm_probe(
    env: gym.Env,
    episodes: int,
) -> List[Trajectory]:
    dummy_action = default_idle_action(env)
    policy = constant_policy(dummy_action)

    def reset_hook(target_env: gym.Env) -> None:
        ensure_idm_vehicle(target_env)

    return _run_probe(
        env=env,
        policy=policy,
        episodes=episodes,
        reset_hook=reset_hook,
    )


def _run_probe(
    env: gym.Env,
    policy: PolicyFn,
    episodes: int,
    *,
    reset_hook: Optional[Callable[[gym.Env], None]] = None,
) -> List[Trajectory]:
    base_seed = int(1e6)
    trajectories: List[Trajectory] = []
    for episode in range(episodes):
        seed = base_seed + episode
        obs, info = env.reset(seed=seed)
        if reset_hook is not None:
            reset_hook(env)

        traj: Trajectory = []
        done = False
        while not done:
            action = policy(obs, info)
            new_obs, reward, terminated, truncated, info = env.step(action)
            step_info = StepInfo(obs, action, reward, new_obs, terminated, truncated, info)
            traj.append(step_info)
            
            obs = new_obs
            done = terminated or truncated
        trajectories.append(traj)

    return trajectories

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


def _collect_env_features(env_name: str, env: gym.Env) -> Dict[str, float]:
    base_env = env.unwrapped
    config = base_env.config
    features: Dict[str, float] = {}

    def add(name: str, value: Any) -> None:
        numeric = as_float(value)
        if numeric is not None:
            features[name] = numeric

    # roundabout and merge don't have these parameters (their values are hardcoded to these)
    lanes = config.get("lanes_count", 2)
    vehicles_count = config.get("vehicles_count", 4)
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

    # obs_space = env.observation_space
    # assert(isinstance(obs_space, gym.spaces.Box))
    # add("observation_space_dim", int(np.prod(obs_space.shape)))

    return features

def _random_probe_features(probe: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    keep_keys = [
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
        "corr_reward_speed",
        "corr_reward_lane_change"
        "collision_observed",
        "timeout_observed",
    ]
    for key in keep_keys:
        _add_prefixed_feature(features, "random", key, probe.get(key))

    mean_return = probe.get("mean_episode_return")
    std_return = probe.get("std_episode_return")
    mean_length = probe.get("mean_episode_length")
    mean_speed = probe.get("mean_speed")
    std_speed = probe.get("std_speed")

    if mean_return is not None and mean_length is not None and mean_length > 0.0:
        features["random_return_per_step"] = mean_return / max(mean_length, 1e-6)
    # if mean_return is not None and std_return is not None:
    #     features["random_reward_snr"] = mean_return / (abs(std_return) + 1e-6)
    # if mean_speed is not None and std_speed is not None:
    #     denom = max(abs(mean_speed), 1e-6)
    #     features["random_speed_cv"] = std_speed / denom

    return features


def _baseline_probe_features(probe: Dict[str, Any], prefix: str) -> Dict[str, float]:
    features: Dict[str, float] = {}
    keep_keys = [
        "mean_episode_return",
        "std_episode_return",
        "episode_length",
        "std_episode_length",
        "collision_rate",
        "timeout_rate",
        "mean_speed",
        "std_speed",
        "collision_observed",
        "timeout_observed",
        "reward_autocorr1",
        "corr_reward_speed",
        "corr_reward_lane_change"
    ]
    for key in keep_keys:
        _add_prefixed_feature(features, prefix, key, probe.get(key))

    mean_speed = probe["mean_speed"]
    std_speed = probe["std_speed"]
    denom = max(abs(mean_speed), 1e-6)
    # features[f"{prefix}_speed_cv"] = std_speed / denom
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

def _derived_features(
    random_probe: Dict[str, Any],
    idm_probe: Dict[str, Any],
) -> Dict[str, float]:
    features: Dict[str, float] = {}

    r_return = random_probe["mean_episode_return"]
    i_return = idm_probe["mean_episode_return"]

    r_std = random_probe["std_episode_return"]
    i_std = idm_probe["std_episode_return"]
    r_speed = random_probe["mean_speed"]
    i_speed = idm_probe["mean_speed"]

    r_collision = random_probe["collision_rate"]
    i_collision = idm_probe["collision_rate"]

    # r_timeout = random_probe["timeout_rate"]
    # i_timeout = idm_probe["timeout_rate"]

    features["idm_return_gain"] = i_return - r_return
    # features["idm_return_ratio"] = i_return / (abs(r_return) + 1e-6)
    features["idm_collision_reduction"] = max(r_collision - i_collision, 0.0)
    features["idm_speed_gain"] = i_speed - r_speed
    features["stability_gap"] = r_std - i_std
    # features["obs_complexity"] = random_probe["obs_flat_dim"]
    features["safety_gap"] = (1.0 - i_collision) - (1.0 - r_collision)
    return features

class BaseMetricHook:
    def on_probe_start(self) -> None:
        pass

    def on_episode_start(self) -> None:
        pass

    def on_step(self, context: StepInfo) -> None:
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

    def on_step(self, context: StepInfo) -> None:
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
        self._lane_change_flags: List[float] = []
        self._progress_deltas: List[float] = []
        self._episode_sequences: List[List[float]] = []
        self._prev_lane: Optional[int] = None
        self._prev_progress: Optional[float] = None

    def on_episode_start(self) -> None:
        self._prev_lane = None
        self._prev_progress = None

    def on_step(self, context: StepInfo) -> None:
        reward = float(context.reward)
        self._rewards.append(reward)

        speed = context.info["speed"]
        self._speed_values.append(float(speed))

        base_env = context.env.unwrapped
        vehicle = base_env.vehicle
        lane_id = _lane_id_from_vehicle(vehicle) if vehicle is not None else None
        if lane_id is not None:
            if self._prev_lane is None:
                lane_change_flag = 0.0
            else:
                lane_change_flag = 1.0 if lane_id != self._prev_lane else 0.0
            self._lane_change_flags.append(lane_change_flag)
            self._prev_lane = lane_id

        progress = None
        pos = _vehicle_position(vehicle)
        if pos.size:
            progress = float(pos[0])
        if progress is not None:
            if self._prev_progress is not None:
                delta = progress - self._prev_progress
                self._progress_deltas.append(delta)
            self._prev_progress = progress

    def finalize(self) -> Dict[str, float]:
        rewards = np.asarray(self._rewards, dtype=float)
        stats: Dict[str, float] = {}
        abs_rewards = np.abs(rewards)
        stats["reward_sparsity"] = float(np.mean(abs_rewards < self.sparsity_epsilon))

        mean_reward = float(np.mean(rewards))
        centered = rewards - mean_reward
        std_reward = float(np.std(rewards))
        # if std_reward > 1e-8:
        #     stats["reward_skew"] = float(np.mean(centered ** 3) / (std_reward ** 3))
        #     stats["reward_kurtosis"] = float(np.mean(centered ** 4) / (std_reward ** 4))
        # else:
        #     stats["reward_skew"] = 0.0
        #     stats["reward_kurtosis"] = 0.0

        autocorr = _safe_autocorr(rewards)
        if autocorr is not None:
            stats["reward_autocorr1"] = autocorr

        corr_speed = _safe_corr(self._rewards, self._speed_values)
        if corr_speed is not None:
            stats["corr_reward_speed"] = corr_speed

        corr_lane = _safe_corr(self._rewards, self._lane_change_flags)
        if corr_lane is not None:
            stats["corr_reward_lane_change"] = corr_lane

        corr_progress = _safe_corr(self._rewards, self._progress_deltas)
        if corr_progress is not None:
            stats["corr_reward_progress"] = corr_progress

        return stats



def _build_metric_hooks() -> List[BaseMetricHook]:
    return [
        # TrafficMetricHook(),
        RewardSignalHook(),
    ]


def _traffic_snapshot(
    context: StepInfo,
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
        other_speed = other.speed
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
    return np.asarray(vehicle.position, dtype=float)


def _lane_id_from_vehicle(vehicle: Any) -> Optional[int]:
    if vehicle is None:
        return None
    lane_index = getattr(vehicle, "lane_index", None)
    if isinstance(lane_index, (list, tuple)) and lane_index:
        lane_index = lane_index[-1]
    if isinstance(lane_index, (np.integer, int)):
        return int(lane_index)
    return None


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

