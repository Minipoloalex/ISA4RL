import numpy as np
import matplotlib.pyplot as plt
import yaml

import itertools
from typing import List, Dict, Any, Iterable, Tuple, Sequence
from pathlib import Path
from pprint import pprint
import time

from utils import (
    save_json,
    read_json,
    CONFIG_FILE,
    BASE_CONFIG_PATH,
    ENV_CONFIG_PATH,
    ALGO_CONFIG_PATH,
    OBS_CONFIG_PATH,
    ALGO_CONFIG_HYPERPARAMS_PATH,
    BASE_IMAGES_PATH,
    ensure_dir,
)

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
    "cartpole-v1",
    "acrobot-v1",
    "mountaincar-v0",
    "atari",
}
ALGO_FILES = ["a2c", "ppo", "dqn"]
ALGO_KEYS_TO_DROP = ["n_timesteps", "normalize", "frame_stack"]

INTERMEDIATE_CONFIG_TYPE = Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]


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


def extract_numeric_matrix(configs: Sequence[Dict[str, Any]]) -> np.ndarray:
    rows = []
    for cfg in configs:
        params = cfg["config"]
        rows.append([params[key] for key in CORRELATION_KEYS])
    return np.asarray(rows, dtype=np.float64)


def save_correlation_plot(configs: Sequence[Dict[str, Any]], output_path: Path) -> None:
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


def build_env_configs() -> List[Dict[str, Any]]:
    """Generate a diverse collection of highway-env configurations.

    The sampler covers a wide spectrum of lane counts, traffic densities, and
    vehicle counts while preserving plausible ratios between them.
    """

    configs: List[Dict[str, Any]] = []
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
                    config = {
                        "name": f"L{lanes}_V{vehicles}_D{str(density).replace('.', 'p')}",
                        "env_id": "highway-v0",
                        "config": {
                            "lanes_count": lanes,
                            "vehicles_count": vehicles,
                            "vehicles_density": density,
                            "duration": duration,
                            "ego_spacing": choose_ego_spacing(density),
                        },
                    }
                    configs.append(config)

    configs.sort(
        key=lambda cfg: (
            cfg["config"]["lanes_count"],
            cfg["config"]["vehicles_count"],
            cfg["config"]["vehicles_density"],
        )
    )
    return configs


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
            configs.append({
                "algo": algo,
                **hyperparams,
            })
    return configs


def get_obs_configs():
    return read_json(OBS_CONFIG_PATH)


def get_env_configs():
    return read_json(ENV_CONFIG_PATH)


def build_all_configs(
    env_configs: List[Dict[str, Any]],
    obs_configs: List[Dict[str, Any]],
    algo_configs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    def conv_config(config: INTERMEDIATE_CONFIG_TYPE) -> Dict[str, Any]:
        env_config, obs_config, algo_config = config
        return {
            "env_config": env_config,
            "obs_config": obs_config,
            "algo_config": algo_config,
            "timestamp": time.time_ns(),
        }
    return list(map(conv_config, itertools.product(env_configs, obs_configs, algo_configs)))

def get_all_configs() -> List[Dict[str, Any]]:
    return build_all_configs(get_env_configs(), get_obs_configs(), get_algo_configs())

if __name__ == "__main__":
    env_configs = build_env_configs()
    save_json(ENV_CONFIG_PATH, env_configs)

    print(f"Total environment configs: {len(env_configs)}")

    ensure_dir(BASE_IMAGES_PATH)
    correlation_plot_path = BASE_IMAGES_PATH / CORRELATION_PLOT_FILE
    save_correlation_plot(env_configs, correlation_plot_path)
    print(f"Correlation plot written to {correlation_plot_path}")

    algo_configs = extract_algo_configs()
    obs_configs = get_obs_configs()
    print("\n\nAlgo Configs")
    pprint(algo_configs)
    print("\n\nObservation Configs")
    pprint(obs_configs)


    all_configs = get_all_configs()
    print(f"\n\nTotal number of configs: {len(all_configs)}")
    print("Example config:")
    pprint(all_configs[0])
    
    save_json(Path(CONFIG_FILE), all_configs)
