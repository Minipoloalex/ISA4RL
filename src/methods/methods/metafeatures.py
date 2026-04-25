import math
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, SupportsFloat
import gymnasium as gym
from gymnasium.spaces.utils import flatdim
import numpy as np

from methods.utils.metafeatures.metrics import SimpleEgoMetricsHook, ObsHook
from methods.utils.metafeatures.other_vehicles_hook import OtherVehiclesBehaviorHook
from methods.configs import InstanceConfig
from methods.utils.general_utils import _flatten_obs
from methods.utils.metafeature_utils import (
    Trajectory,
    StepInfo,
    make_random_policy,
    make_parking_geometric_policy,
    constant_policy,
    default_idle_action,
    ensure_idm_vehicle,
    is_parking_env,
    PolicyFn,
    get_action_space_size,
    get_max_episode_steps,
)
from methods.utils.metafeatures.model_based import (
    estimate_normalized_lipschitz,
    compute_transition_stochasticity,
    compute_transition_linearity,
    compute_action_landscape_ruggedness,
    compute_state_entropy,
)

FEATURES_KEY = "features"
TIMESTAMP_KEY = "timestamp"
ELAPSED_TIME_KEY = "elapsed_time"

logger = logging.getLogger(__name__)

def extract_metafeatures(
    config: InstanceConfig,
    requested_groups: Optional[List[str]] = None,
    existing_data: Optional[Dict] = None,
    update_threshold: float = 0.0
):
    if existing_data is None:
        existing_data = {}
        
    out_data = dict(existing_data)
    env = config.ensure_test_env()
    out_data["env_name"] = env.spec.id

    if "feature_groups" not in out_data:
        out_data["feature_groups"] = {}
        
    def should_compute(g: str) -> bool:
        if requested_groups is not None and g not in requested_groups:
            return False
        if out_data["feature_groups"].get(g, {}).get("timestamp", 0.0) > update_threshold:
            return False
        return True


    if should_compute("env_features"):
        logger.info(f"Computing env_features for instance: {config.instance_folder_path}")
        before = time.perf_counter()
        env_features = _collect_env_features(out_data["env_name"], env)
        elapsed = time.perf_counter() - before
        out_data["feature_groups"]["env_features"] = {
            TIMESTAMP_KEY: time.time(),
            ELAPSED_TIME_KEY: elapsed,
            FEATURES_KEY: env_features,
        }

    if should_compute("probes"):
        logger.info(f"Computing probes for instance: {config.instance_folder_path}")
        before = time.perf_counter()
        trajectories_random = _run_probe(
            env=env,
            policy=make_random_policy(env),
            episodes=config.n_test_episodes,
        )
        trajectories_baseline = _run_baseline_probe(
            env=env,
            episodes=config.n_test_episodes,
        )
        random_probe = traj_metafeatures(trajectories_random)
        baseline_probe = traj_metafeatures(trajectories_baseline)
        logger.info(f"Random probe achieved performance of {random_probe["reward_mean"]} mean return")
        logger.info(f"Structured probe achieved performance of {baseline_probe["reward_mean"]} mean return")
        # idm_advantage = idm_probe.get("mean_episode_return", 0.0) - random_probe.get("mean_episode_return", 0.0)
        # safety_delta = random_probe.get("collision_rate", 0.0) - idm_probe.get("collision_rate", 0.0)
        
        features: Dict[str, float] = {}
        features.update(_probe_features(random_probe, "random"))
        features.update(_probe_features(baseline_probe, "baseline"))
        # Backward-compatible aliases for existing analysis scripts. For
        # parking-v0 these represent the geometric parking baseline, not IDM.
        # features.update(
        #     _combine_probe_features(
        #         random_probe=random_probe,
        #         idm_probe=idm_probe,
        #     )
        # )
        
        # Also compute diagnostics here since they are probe-derived
        diagnostics = {
            # "idm_advantage": idm_advantage,
            # "idm_safety_gain": safety_delta,
        }
        
        elapsed = time.perf_counter() - before
        out_data["feature_groups"]["probes"] = {
            TIMESTAMP_KEY: time.time(),
            ELAPSED_TIME_KEY: elapsed,
            FEATURES_KEY: features,
            "diagnostics": diagnostics,
            "probes_raw": {
                "random": random_probe,
                "baseline": baseline_probe,
            }
        }

    mb_funcs = {
        "mb_normalized_lipschitz": estimate_normalized_lipschitz,
        "mb_transition_stochasticity": compute_transition_stochasticity,
        "mb_transition_linearity": compute_transition_linearity,
        "mb_action_landscape_ruggedness": compute_action_landscape_ruggedness,
        "mb_state_entropy": compute_state_entropy,
    }

    for mb_group_name, func in mb_funcs.items():
        if should_compute(mb_group_name):
            logger.info(f"Computing {mb_group_name} for instance: {config.instance_folder_path}")
            before = time.perf_counter()
            val = func(env)
            elapsed = time.perf_counter() - before
            feature_name = mb_group_name.replace("mb_", "")
            out_data["feature_groups"][mb_group_name] = {
                TIMESTAMP_KEY: time.time(),
                ELAPSED_TIME_KEY: elapsed,
                FEATURES_KEY: {feature_name: val},
            }
        else:
            logger.info(f"Skipping {mb_group_name} for instance: {config.instance_folder_path}")
    logger.info(f"Done computing MB features for instance: {config.instance_folder_path}")

    # Takes too long to compute
    # if should_compute("pic"):
    #     logger.info(f"Computing pic for instance: {config.instance_folder_path}")
    #     from methods.utils.metafeatures.pic_metafeature import compute_pic_end_to_end
        
    #     before = time.perf_counter()
    #     pic_val, poic_val, metrics_dict, _ = compute_pic_end_to_end(
    #         env=env,
    #         n_samples=1000,
    #         n_episodes=10,
    #         env_name=out_data["env_name"],
    #     )
    #     elapsed = time.perf_counter() - before
        
    #     features = {
    #         "pic": float(pic_val),
    #         "poic": float(poic_val),
    #     }
    #     for k, v in metrics_dict.items():
    #         features[k] = float(v)
            
    #     out_data["feature_groups"]["pic"] = {
    #         TIMESTAMP_KEY: time.time(),
    #         ELAPSED_TIME_KEY: elapsed,
    #         FEATURES_KEY: features,
    #     }
    config.close()
    return out_data


def traj_metafeatures(trajectories: List[Trajectory]) -> Dict[str, Any]:
    # Initialize the hook we built earlier
    hooks = [SimpleEgoMetricsHook(), ObsHook(), OtherVehiclesBehaviorHook()]
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
    metrics = {}
    for hook in hooks:
        metrics.update(hook.finalize())

    return metrics

def _run_baseline_probe(
    env: gym.Env,
    episodes: int,
) -> List[Trajectory]:
    if is_parking_env(env):
        return _run_parking_geometric_probe(env, episodes)
    return _run_idm_probe(env, episodes)


def _run_parking_geometric_probe(
    env: gym.Env,
    episodes: int,
) -> List[Trajectory]:
    return _run_probe(
        env=env,
        policy=make_parking_geometric_policy(env),
        episodes=episodes,
    )


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

            info = dict(info)
            # Save traffic-derived signals into info so hooks can use them later.
            info["other_vehicles_states"] = _extract_other_vehicles_states(env)
            min_ttc = _extract_min_ttc(env)
            if min_ttc is not None:
                info["min_ttc"] = min_ttc

            step_info = StepInfo(obs, action, reward, new_obs, terminated, truncated, info)
            traj.append(step_info)
            
            obs = new_obs
            done = terminated or truncated
        trajectories.append(traj)

    return trajectories

def _extract_other_vehicles_states(env: gym.Env) -> List[Dict[str, float]]:
    base_env = env.unwrapped
    
    # Try Highway-env
    if hasattr(base_env, "road") and hasattr(base_env.road, "vehicles"):
        ego = getattr(base_env, "vehicle", None)
        states = []
        for v in base_env.road.vehicles:
            if v is ego: continue
            states.append({
                "id": id(v),
                "speed": float(v.speed),
                "heading": float(v.heading),
            })
        return states

    # Try Metadrive
    if hasattr(base_env, "engine"):
        ego = getattr(base_env, "vehicle", None)
        states = []
        tm = getattr(base_env.engine, "traffic_manager", None)
        if tm is not None:
            vehicles = getattr(tm, "vehicles", [])
            for v in vehicles:
                if v is ego: continue
                states.append({
                    "id": id(v),
                    "speed": float(getattr(v, "speed", 0.0)),
                    "heading": float(getattr(v, "heading_theta", getattr(v, "heading", 0.0))),
                })
        return states
        
    return []


def _extract_min_ttc(env: gym.Env) -> Optional[float]:
    base_env = env.unwrapped
    road = getattr(base_env, "road", None)
    ego = getattr(base_env, "vehicle", None)
    if road is None or ego is None:
        return None

    vehicles = getattr(road, "vehicles", None)
    if vehicles is None:
        return None

    ego_lane = _lane_id_from_vehicle(ego)
    ego_pos = _vehicle_position(ego)
    ego_speed = getattr(ego, "speed", None)
    if ego_lane is None or ego_pos.size < 2 or ego_speed is None:
        return None

    ttc_samples: List[float] = []
    for other in vehicles:
        if other is ego:
            continue

        other_lane = _lane_id_from_vehicle(other)
        if other_lane is None or abs(other_lane - ego_lane) > 1:
            continue

        other_pos = _vehicle_position(other)
        other_speed = getattr(other, "speed", None)
        if other_pos.size < 2 or other_speed is None:
            continue

        longitudinal_gap = float(other_pos[0] - ego_pos[0])
        if longitudinal_gap >= 0.0:
            ttc = _ttc_from_gap(longitudinal_gap, _closing_speed(float(ego_speed), float(other_speed)))
        else:
            ttc = _rear_ttc(abs(longitudinal_gap), float(other_speed), float(ego_speed))
        ttc_samples.append(ttc)

    finite_ttc = [value for value in ttc_samples if math.isfinite(value)]
    return float(min(finite_ttc)) if finite_ttc else float("inf")


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

def _collect_env_features(env_name: str, env: gym.Env) -> Dict[str, float]:
    base_env = env.unwrapped
    config = base_env.config
    features: Dict[str, float] = {}

    lanes = config.get("lanes_count") or config.get("roundabout_lanes")
    traffic_density = config.get("vehicles_density") or 1
    vehicles_count = config.get("vehicles_count") or 0

    features["lanes_count"] = lanes
    features["traffic_density"] = traffic_density
    features["action_space_size"] = get_action_space_size(env)
    features["max_steps"] = get_max_episode_steps(env)

    obs_space = env.observation_space
    
    # Optional fallback for original log dim just in case
    # If space is Dict/Tuple, np.prod doesn't work directly, so we flatten it
    try:
        dim = flatdim(obs_space)
        features["obs_space_dim_log"] = int(np.log2(dim)) if dim > 0 else 0
    except Exception:
        if isinstance(obs_space, gym.spaces.Box):
            features["obs_space_dim_log"] = int(np.log2(np.prod(obs_space.shape)))
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")

    cat_count, num_count, bin_count = _count_space_attributes(obs_space)
    features["obs_categorical_count"] = float(cat_count)
    features["obs_numerical_count"] = float(num_count)
    features["obs_binary_count"] = float(bin_count)

    return features

def _count_space_attributes(space: gym.Space) -> Tuple[int, int, int]:
    num_cat = 0
    num_num = 0
    num_bin = 0

    if isinstance(space, gym.spaces.Box):
        is_int = np.issubdtype(space.dtype, np.integer)
        flat_low = space.low.flatten()
        flat_high = space.high.flatten()
        for l, h in zip(flat_low, flat_high):
            if is_int:
                if l == 0 and h == 1:
                    num_bin += 1
                else:
                    num_cat += 1
            else:
                num_num += 1
    elif isinstance(space, gym.spaces.Discrete):
        if space.n == 2:
            num_bin += 1
        else:
            num_cat += 1
    elif isinstance(space, gym.spaces.MultiDiscrete):
        for n in space.nvec.flatten():
            if n == 2:
                num_bin += 1
            else:
                num_cat += 1
    elif isinstance(space, gym.spaces.MultiBinary):
        num_bin += int(np.prod(space.shape))
    elif isinstance(space, gym.spaces.Tuple):
        for s in space.spaces:
            c, n, b = _count_space_attributes(s)
            num_cat += c
            num_num += n
            num_bin += b
    elif isinstance(space, gym.spaces.Dict):
        for s in space.spaces.values():
            c, n, b = _count_space_attributes(s)
            num_cat += c
            num_num += n
            num_bin += b

    return num_cat, num_num, num_bin

def _probe_features(probe: Dict[str, Any], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{key}": probe[key] for key in probe}

# def _combine_probe_features(
#     random_probe: Dict[str, Any],
#     idm_probe: Dict[str, Any],
# ) -> Dict[str, float]:
#     features: Dict[str, float] = {}

#     r_return = random_probe["mean_episode_return"]
#     i_return = idm_probe["mean_episode_return"]

#     r_std = random_probe["std_episode_return"]
#     i_std = idm_probe["std_episode_return"]
#     r_speed = random_probe["mean_speed"]
#     i_speed = idm_probe["mean_speed"]

#     r_collision = random_probe["collision_rate"]
#     i_collision = idm_probe["collision_rate"]

#     # r_timeout = random_probe["timeout_rate"]
#     # i_timeout = idm_probe["timeout_rate"]

#     features["idm_return_gain"] = i_return - r_return
#     # features["idm_return_ratio"] = i_return / (abs(r_return) + 1e-6)
#     features["idm_collision_reduction"] = max(r_collision - i_collision, 0.0)
#     features["idm_speed_gain"] = i_speed - r_speed
#     features["stability_gap"] = r_std - i_std
#     # features["obs_complexity"] = random_probe["obs_flat_dim"]
#     features["safety_gap"] = (1.0 - i_collision) - (1.0 - r_collision)
#     return features
