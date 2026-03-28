import numpy as np
import matplotlib.pyplot as plt
import yaml

import itertools
from typing import List, Dict, Any, Iterable, Tuple, Sequence, Optional, Callable
from pathlib import Path
from pprint import pprint
import time
from copy import deepcopy

from common.config_utils import ENVS, CONFIG
from common.file_utils import (
    save_json,
    read_json,
    ensure_dir,
    TRAIN_CONFIGS_PATH,
    EVAL_CONFIGS_PATH,
    ENV_CONFIG_PATH,
    ALGO_CONFIG_PATH,
    OBS_CONFIG_PATH,
    ALGO_CONFIG_HYPERPARAMS_PATH,
    BASE_IMAGES_PATH,
)
from env_fixed_configs import *

# Config generation parameters
MIN_LANE_COUNT = 2
MAX_LANE_COUNT = 5
LANE_CAPACITY_RANGE = (12.0, 18.0)  # approximate per-lane throughput for min/max lanes

MIN_VEHICLE_COUNT = 0
MAX_VEHICLE_COUNT = 120
MIN_NONZERO_VEHICLES_PER_LANE = 2
MAX_VEHICLES_PER_LANE = 18
VEHICLE_COUNT_STEP = 5
VEHICLE_COUNT_VARIATIONS = (-10, -5, 0, 5, 10, 15, 20)

OCCUPANCY_TARGETS = (0.0, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15)
OCCUPANCY_DENSITY_BANDS: Tuple[Tuple[float, float, Sequence[float]], ...] = (
    (0.0, 0.1, (0.45, 0.5, 0.55)),
    (0.1, 0.4, (0.6, 0.7, 0.8, 0.9)),
    (0.4, 0.65, (0.95, 1.05, 1.15)),
    (0.65, 0.85, (1.2, 1.3, 1.4, 1.5)),
    (0.85, 1.05, (1.45, 1.6, 1.7, 1.8)),
    (1.05, 1.3, (1.75, 1.85, 1.95)),
)
MIN_DENSITY = 0.45
MAX_DENSITY = 2.0

BASE_DURATION = 35
LANE_DURATION_BONUS = 3
OCCUPANCY_DURATION_PIVOT = 0.6
OCCUPANCY_DURATION_GAIN = 18
LOW_DENSITY_THRESHOLD = 0.85
LOW_DENSITY_BONUS = 6
HIGH_DENSITY_THRESHOLD = 1.5
HIGH_DENSITY_PENALTY = 8
SMALL_FLEET_SCALE = 7
DENSE_SMALL_FLEET_CAP = 30
MIN_DURATION = 20
MAX_DURATION = 75
DURATION_ROUNDING = 5

EGO_SPACING_BASE = 3.2
EGO_SPACING_SLOPE = 0.9
EGO_SPACING_MIN = 1.5
EGO_SPACING_MAX = 4.5

CORRELATION_PLOT_FILE = "config_correlations.png"
CORRELATION_KEYS = (
    "lanes_count",
    "vehicles_count",
    "vehicles_density",
    "duration",
    "ego_spacing",
)

ALGO_HYPERPAMETER_ENVS = {
    "lunarlander-v3",
    "atari",
}
ALGO_FILES = ["a2c", "ppo", "dqn"]
ALGO_KEYS_TO_DROP = ["n_timesteps", "normalize", "frame_stack", "env_wrapper"]

OBS_CNN = ["GrayscaleObservation"]
OBS_MLP = ["Kinematics", "TimeToCollision"]

EVAL_BASE_SEED = int(1e6)
EVAL_SEED_COUNT = {
    "highway": 1,
    "merge": 50,
    "roundabout": 50,
}


def lane_capacity(lanes: int) -> float:
    return float(
        np.interp(lanes, (MIN_LANE_COUNT, MAX_LANE_COUNT), LANE_CAPACITY_RANGE)
    )


def round_to_step(value: float) -> int:
    if value <= 0:
        return 0
    return int(VEHICLE_COUNT_STEP * max(1, round(value / VEHICLE_COUNT_STEP)))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def vehicle_counts_for_occupancy(
    lanes: int, capacity: float, target_occupancy: float
) -> Iterable[int]:
    if target_occupancy <= 0:
        yield 0
        return

    base = lanes * capacity * target_occupancy
    candidate_counts = set()

    for offset in VEHICLE_COUNT_VARIATIONS:
        count = round_to_step(base + offset)
        if count == 0:
            continue
        if count < MIN_VEHICLE_COUNT or count > MAX_VEHICLE_COUNT:
            continue
        per_lane = count / lanes
        if per_lane < MIN_NONZERO_VEHICLES_PER_LANE or per_lane > MAX_VEHICLES_PER_LANE:
            continue
        candidate_counts.add(count)

    if not candidate_counts:
        count = round_to_step(base)
        if 0 < count <= MAX_VEHICLE_COUNT:
            per_lane = count / lanes if lanes else 0
            if MIN_NONZERO_VEHICLES_PER_LANE <= per_lane <= MAX_VEHICLES_PER_LANE:
                candidate_counts.add(count)

    for count in sorted(candidate_counts):
        yield count


def density_options_for_occupancy(occupancy: float) -> Sequence[float]:
    for lower, upper, options in OCCUPANCY_DENSITY_BANDS:
        if lower <= occupancy < upper:
            return tuple(d for d in options if MIN_DENSITY <= d <= MAX_DENSITY)
    fallback_options = OCCUPANCY_DENSITY_BANDS[-1][2]
    return tuple(d for d in fallback_options if MIN_DENSITY <= d <= MAX_DENSITY)


def choose_duration(lanes: int, vehicles: int, density: float, capacity: float) -> int:
    occupancy = vehicles / (lanes * capacity) if lanes and capacity else 0.0
    base = BASE_DURATION + LANE_DURATION_BONUS * (lanes - MIN_LANE_COUNT)
    base += int(OCCUPANCY_DURATION_GAIN * (occupancy - OCCUPANCY_DURATION_PIVOT))
    if density <= LOW_DENSITY_THRESHOLD:
        base += LOW_DENSITY_BONUS
    elif density >= HIGH_DENSITY_THRESHOLD:
        base -= HIGH_DENSITY_PENALTY
    if vehicles <= lanes * SMALL_FLEET_SCALE and density >= HIGH_DENSITY_THRESHOLD:
        base = min(base, DENSE_SMALL_FLEET_CAP)
    base = clamp(base, MIN_DURATION, MAX_DURATION)
    return int(DURATION_ROUNDING * round(base / DURATION_ROUNDING))


def choose_ego_spacing(density: float) -> float:
    spacing = EGO_SPACING_BASE - (density - 1.0) * EGO_SPACING_SLOPE
    return float(round(clamp(spacing, EGO_SPACING_MIN, EGO_SPACING_MAX), 1))


def extract_numeric_matrix(configs: Sequence[CONFIG]) -> np.ndarray:
    rows = []
    for cfg in configs:
        params = cfg["config"]
        rows.append([params[key] for key in CORRELATION_KEYS])
    return np.asarray(rows, dtype=np.float64)


def save_correlation_plot(configs: Sequence[CONFIG], output_path: Path) -> None:
    data = extract_numeric_matrix(configs)
    if data.shape[0] < 2:
        raise ValueError(
            "At least two configurations are required to compute correlations."
        )
    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(CORRELATION_KEYS)))
    ax.set_xticklabels(CORRELATION_KEYS, rotation=45, ha="right")
    ax.set_yticks(range(len(CORRELATION_KEYS)))
    ax.set_yticklabels(CORRELATION_KEYS)
    ax.set_title("Correlation between highway-env config variables")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_highway_configs() -> List[CONFIG]:
    """Generate a diverse collection of highway-env configurations.

    The sampler covers a wide spectrum of lane counts, traffic densities, and
    vehicle counts while preserving plausible ratios between them.
    """

    configs: List[CONFIG] = []
    dedup: set[Tuple[int, int, float]] = set()

    for lanes in range(MIN_LANE_COUNT, MAX_LANE_COUNT + 1):
        capacity = lane_capacity(lanes)

        for target_occupancy in OCCUPANCY_TARGETS:
            for vehicles in vehicle_counts_for_occupancy(
                lanes, capacity, target_occupancy
            ):
                if vehicles > MAX_VEHICLE_COUNT:
                    continue

                occupancy = vehicles / (lanes * capacity) if lanes and capacity else 0.0
                density_candidates = density_options_for_occupancy(occupancy)

                for density in density_candidates:
                    density = float(round(density, 2))
                    if not (MIN_DENSITY <= density <= MAX_DENSITY):
                        continue
                    if vehicles == 0 and density > MIN_DENSITY:
                        continue

                    key = (lanes, vehicles, density)
                    if key in dedup:
                        continue
                    dedup.add(key)

                    duration = choose_duration(lanes, vehicles, density, capacity)

                    config = deepcopy(HIGHWAY_FIXED_CONFIGS)
                    config["config"].update({
                            "lanes_count": lanes,
                            "vehicles_count": vehicles,
                            "vehicles_density": density,
                            "duration": duration,
                            "ego_spacing": choose_ego_spacing(density),
                    })
                    configs.append(config)

    configs.sort(
        key=lambda cfg: (
            cfg["config"]["lanes_count"],
            cfg["config"]["vehicles_count"],
            cfg["config"]["vehicles_density"],
        )
    )
    return configs

def build_merge_configs() -> List[CONFIG]:
    return [
        deepcopy(MERGE_FIXED_CONFIGS)
    ]

def build_roundabout_configs() -> List[CONFIG]:
    return [
        deepcopy(ROUNDABOUT_FIXED_CONFIGS)
    ]

def build_u_turn_configs() -> List[CONFIG]:
    return [
        deepcopy(U_TURN_FIXED_CONFIGS)
    ]

def build_two_way_configs() -> List[CONFIG]:
    return [
        deepcopy(TWO_WAY_FIXED_CONFIGS)
    ]

def build_exit_configs() -> List[CONFIG]:
    raise NotImplementedError

def build_lane_keeping_configs() -> List[CONFIG]:
    raise NotImplementedError

def build_racetrack_configs() -> List[CONFIG]:
    raise NotImplementedError

def build_parking_configs() -> List[CONFIG]:
    raise NotImplementedError
    
# def build_seeded_configs(config: CONFIG, base_seed: int, seed_cnt: int) -> List[CONFIG]:
#     seeds = range(base_seed, base_seed + seed_cnt)
#     seeded_configs = [
#         {
#             **config,
#             "eval_seed": seed,
#         }
#         for seed in seeds
#     ]
#     return seeded_configs


def extract_algo_configs():
    aggregated: Dict[str, Dict[str, Any]] = {}

    for algo in ALGO_FILES:
        config_path = ALGO_CONFIG_HYPERPARAMS_PATH / f"{algo}.yml"
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}
        if not isinstance(raw_config, dict):
            raise ValueError(
                f"Unexpected structure in {config_path}: expected a mapping."
            )

        filtered = {
            env_name: params
            for env_name, params in raw_config.items()
            if isinstance(env_name, str) and env_name.lower() in ALGO_HYPERPAMETER_ENVS
        }
        if filtered:
            aggregated[algo] = filtered

    ALGO_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(ALGO_CONFIG_PATH, aggregated)
    return aggregated


def get_algo_configs():
    algo_configs = read_json(ALGO_CONFIG_PATH)
    configs = []
    for algo, hyperparam_config in algo_configs.items():
        for env, hyperparams in hyperparam_config.items():
            for key in ALGO_KEYS_TO_DROP:
                hyperparams.pop(key, None)
            configs.append(
                {
                    "algo": algo,
                    **hyperparams,
                }
            )
    return configs

def get_obs_configs():
    return read_json(OBS_CONFIG_PATH)

def get_env_configs(env_id: str) -> List[CONFIG]:
    return read_json(ENV_CONFIG_PATH(env_id))

def valid_config(algo_config: CONFIG, obs_config: CONFIG) -> bool:
    return (
        algo_config["policy"] == "CnnPolicy"
        and obs_config["type"] in OBS_CNN
    ) or (
        algo_config["policy"] == "MlpPolicy"
        and obs_config["type"] in OBS_MLP
    )

def build_all_configs(
    env_configs: List[CONFIG],
    obs_configs: List[CONFIG],
    algo_configs: List[CONFIG],
) -> Tuple[List[CONFIG], List[CONFIG]]:
    run_configs = [
        {
            "env_config": config[0],
            "obs_config": config[1],
            "algo_config": config[2],
            "timestamp": time.time_ns(),
        }
        for config in itertools.product(
            env_configs, obs_configs, algo_configs,
        )
        if valid_config(config[2], config[1])
    ]
    eval_configs = [
        {
            "env_config": config[0],
            "obs_config": config[1],
            "timestamp": time.time_ns(),
        }
        for config in itertools.product(
            env_configs, obs_configs,
        )
    ]
    return run_configs, eval_configs

# def build_all_configs(
#     env_configs_train: List[CONFIG],
#     env_configs_eval: List[CONFIG],  # these also include some configurations where only the seed varies
#     obs_configs: List[CONFIG],
#     algo_configs: List[CONFIG],
# ) -> Tuple[List[CONFIG], List[CONFIG], List[CONFIG]]:
#     def conv_run_config(config: INTERMEDIATE_TRAIN_CONFIG) -> CONFIG:
#         env_config, obs_config, algo_config = config
#         return {
#             "env_config": env_config,
#             "obs_config": obs_config,
#             "algo_config": algo_config,
#             "timestamp": time.time_ns(),
#         }
#     def conv_instance_config(config: INTERMEDIATE_EVAL_CONFIG) -> CONFIG:
#         env_config, obs_config = config
#         return {
#             "env_config": env_config,
#             "obs_config": obs_config,
#             "timestamp": time.time_ns(),
#         }

#     run_configs = list(
#         filter(
#             valid_config,
#             map(
#                 conv_run_config,
#                 itertools.product(env_configs_train, obs_configs, algo_configs),
#             ),
#         )
#     )
#     eval_configs = list(
#         filter(
#             valid_config,
#             map(
#                 conv_run_config,
#                 itertools.product(env_configs_eval, obs_configs, algo_configs),
#             ),
#         )
#     )
#     instance_configs = list(
#         map(
#             conv_instance_config,
#             itertools.product(env_configs_eval, obs_configs),
#         )
#     )
#     run_configs = annotate_ids(run_configs)
#     eval_configs = annotate_ids(eval_configs)
#     instance_configs = annotate_ids(instance_configs)
#     return run_configs, eval_configs, instance_configs


# def get_all_configs() -> Tuple[List[CONFIG], List[CONFIG], List[CONFIG]]:
#     env_configs_train = [
#         build_seeded_configs(config, EVAL_BASE_SEED, 1)[0]
#         for config in
#         get_highway_configs() + get_roundabout_configs() + get_merge_configs()
#     ]
#     env_configs_train = annotate_ids(env_configs_train)
#     for cfg in env_configs_train:
#         cfg["orig_id"] = cfg["id"]

#     env_configs_eval = []
#     for train_config in env_configs_train:
#         env_name = train_config["env_id"].split("-")[0]
#         seed_configs = build_seeded_configs(train_config, EVAL_BASE_SEED, EVAL_SEED_COUNT[env_name])
#         for conf in seed_configs:
#             conf["orig_id"] = train_config["orig_id"]
#         env_configs_eval.extend(seed_configs)

#     env_configs_eval = annotate_ids(env_configs_eval)
#     return build_all_configs(env_configs_train, env_configs_eval, get_obs_configs(), get_algo_configs())

def build_configs(builder: Callable[[], List[CONFIG]], save_path: Path, name: str) -> List[CONFIG]:
    configs = builder()
    save_json(save_path, configs)
    print(f"{name} environment configs: {len(configs)}")
    return configs

if __name__ == "__main__":
    highway_configs = build_configs(build_highway_configs, ENV_CONFIG_PATH("highway-fast-v0"), "Highway")
    build_configs(build_merge_configs, ENV_CONFIG_PATH("merge-generic-v0"), "Merge")
    build_configs(build_roundabout_configs, ENV_CONFIG_PATH("roundabout-generic-v0"), "Roundabout")

    ensure_dir(BASE_IMAGES_PATH)
    correlation_plot_path = BASE_IMAGES_PATH / CORRELATION_PLOT_FILE
    save_correlation_plot(highway_configs, correlation_plot_path)
    print(f"Correlation plot written to {correlation_plot_path}")

    algo_configs = extract_algo_configs()
    obs_configs = get_obs_configs()
    print(f"\n\nAlgo Configs: {len(algo_configs)}")
    print(f"\n\nObservation Configs: {len(obs_configs)}")

    all_run_configs = []
    all_eval_configs = []
    for env in ENVS:
        env_run_configs, env_eval_configs = build_all_configs(
            get_env_configs(env), get_obs_configs(), get_algo_configs(),
        )
        save_json(TRAIN_CONFIGS_PATH(env), env_run_configs)
        save_json(EVAL_CONFIGS_PATH(env), env_eval_configs)
        all_run_configs.append(env_run_configs)
        all_eval_configs.append(env_eval_configs)

    print(f"\n\nTotal number of train configs: {len(all_run_configs)}")
    print("Example config:")
    pprint(all_run_configs[0])

    print(f"\n\nTotal number of eval configs: {len(all_eval_configs)}")
    print("Example config:")
    pprint(all_eval_configs[0])
