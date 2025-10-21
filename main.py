"""ISA pipeline for highway-v0 instances.

This module instantiates a set of highway-env configurations, extracts structural
and probe-based meta-features, evaluates a portfolio of RL agents, and prepares
artifacts that can be consumed for instance space analysis.

The implementation follows a compact blueprint:
1. Generate parameter sweeps -> `EnvConfig` objects.
2. Compute structural, probe, and trajectory embedding features.
3. Evaluate a small algorithm portfolio and derive per-instance winners.
4. Save tables and lightweight plots for downstream ISA work.

The code is written to be run as a script:
    python main.py --output-dir outputs/highway --instances 24
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, deque, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from train import train
from utils import (
    set_global_seed,
    ensure_dir,
    discretize,
    _flatten_obs,
    _normalize_action,
    _round_half_up,
    _interpolate_range_value,
    _coerce_numeric,
    _json_default,
)

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required: pip install gymnasium") from exc

try:
    import highway_env  # noqa: F401  # ensures envs are registered
except ImportError as exc:  # pragma: no cover
    raise ImportError("highway-env is required: pip install highway-env") from exc

try:
    from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required for training probes and portfolio agents."
    ) from exc

try:
    from sklearn.linear_model import Ridge
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError("pandas is required: pip install pandas") from exc

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for trajectory embeddings.") from exc


# ---------------------------------------------------------------------------
# Data schemas


EnvMaker = Callable[[], gym.Env]


@dataclass
class EnvConfig:
    env_id: str
    make_fn: EnvMaker
    params: Dict[str, Any]


@dataclass
class StructuralFeatures:
    horizon: int
    early_terminal_rate: float
    random_crash_rate: float
    reward_mean: float
    reward_std: float
    reward_skew: float
    reward_kurtosis: float
    next_state_entropy: float
    branching_factor: float
    observability_gap: float
    transition_stochasticity: float
    reward_sparsity: float
    mean_reward_gap: float
    mean_first_reward_step: float

# @dataclass
# class ConfigMetaFeatures:
#     observation_space_size: int
#     vehicles_count: int
#     lanes_count: int
#     duration: int
#     reward_speed_range: Tuple[int,int]
#     collision_reward_weight: float
#     high_speed_reward_weight: float
#     lane_change_reward_weight: float

@dataclass
class MetaFeatures:
    env_id: str
    structural: StructuralFeatures
    # probe: ProbeFeatures
    domain_params: Dict[str, Any]

# ---------------------------------------------------------------------------
# Structural features


def estimate_structural(
    env: gym.Env,
    num_episodes: int = 50,
    max_steps: int = 300,
    history_k: int = 4,
    bins_per_dim: int = 15,
) -> StructuralFeatures:
    term_early = 0
    crash_count = 0
    rewards: List[float] = []
    next_bin_counts: List[Tuple[int, ...]] = []
    bin_totals: Counter = Counter()
    transition_counts = defaultdict(Counter)
    state_action_counts: Counter = Counter()
    zero_reward_steps = 0
    total_steps = 0
    reward_gap_lengths: List[int] = []
    first_reward_steps: List[int] = []

    X_no_hist: List[np.ndarray] = []
    X_hist: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        state = _flatten_obs(state)
        hist = deque([np.zeros_like(state) for _ in range(history_k - 1)], maxlen=history_k - 1)
        episode_reward = 0.0
        steps_since_reward = 0
        first_reward_observed = False

        for step in range(max_steps):
            state_bin = discretize(state, bins_per_dim=bins_per_dim)
            action = env.action_space.sample()
            action_key = _normalize_action(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = _flatten_obs(next_state)
            done = terminated or truncated

            reward_val = float(reward)
            rewards.append(reward_val)
            episode_reward += reward_val

            total_steps += 1
            if abs(reward_val) < 1e-6:
                zero_reward_steps += 1
                steps_since_reward += 1
            else:
                reward_gap_lengths.append(steps_since_reward)
                steps_since_reward = 0
                if not first_reward_observed:
                    first_reward_steps.append(step + 1)
                    first_reward_observed = True

            binned = discretize(next_state, bins_per_dim=bins_per_dim)
            next_bin_counts.append(binned)
            bin_totals[binned] += 1
            state_action_key = (state_bin, action_key)
            transition_counts[state_action_key][binned] += 1
            state_action_counts[state_action_key] += 1

            X_no_hist.append(state)
            hist_stack = np.concatenate([*hist, state]) if hist else state
            X_hist.append(hist_stack)
            targets.append(next_state)

            if len(hist) == hist.maxlen:
                hist.popleft()
            hist.append(state)
            state = next_state

            if done:
                if step < max_steps - 1:
                    term_early += 1
                crashed = info["crashed"]
                print(f"step: {step}, crashed: {crashed}")
                if crashed:
                    crash_count += 1
                if not first_reward_observed:
                    first_reward_steps.append(max_steps)
                    reward_gap_lengths.append(max_steps)
                break

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    r_mean = float(np.mean(rewards_arr))
    r_std = float(np.std(rewards_arr) + 1e-8)

    if len(rewards_arr) > 2:
        centered = rewards_arr - r_mean
        r_skew = float(np.mean(centered**3) / (r_std**3 + 1e-8))
    else:
        r_skew = 0.0

    if len(rewards_arr) > 3:
        centered = rewards_arr - r_mean
        r_kurt = float(np.mean(centered**4) / (r_std**4 + 1e-8))
    else:
        r_kurt = 0.0

    counts = np.asarray(list(bin_totals.values()), dtype=np.float64)
    probs = counts / (counts.sum() + 1e-12)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    branching_factor = float(len(bin_totals) / (len(next_bin_counts) + 1e-8))

    X0 = np.asarray(X_no_hist, dtype=np.float32)
    Xh = np.asarray(X_hist, dtype=np.float32)
    Y = np.asarray(targets, dtype=np.float32)

    model_no_hist = Ridge(alpha=1.0).fit(X0, Y)
    model_hist = Ridge(alpha=1.0).fit(Xh, Y)
    mse_no_hist = float(np.mean((model_no_hist.predict(X0) - Y) ** 2))
    mse_hist = float(np.mean((model_hist.predict(Xh) - Y) ** 2))
    observability_gap = mse_no_hist - mse_hist

    total_sa_visits = sum(state_action_counts.values())
    if total_sa_visits > 0:
        transition_stochasticity = 0.0
        for key, next_counts in transition_counts.items():
            sa_visits = state_action_counts[key]
            if sa_visits == 0:
                continue
            probs_sa = np.asarray(list(next_counts.values()), dtype=np.float64) / sa_visits
            sa_entropy = float(-(probs_sa * np.log(probs_sa + 1e-12)).sum())
            transition_stochasticity += (sa_visits / total_sa_visits) * sa_entropy
    else:
        transition_stochasticity = 0.0

    reward_sparsity = zero_reward_steps / max(1, total_steps)
    mean_reward_gap = float(np.mean(reward_gap_lengths)) if reward_gap_lengths else float(max_steps)
    mean_first_reward_step = float(np.mean(first_reward_steps)) if first_reward_steps else float(max_steps)

    horizon = getattr(env, "_max_episode_steps", max_steps)

    return StructuralFeatures(
        horizon=int(horizon),
        early_terminal_rate=term_early / max(1, num_episodes),
        random_crash_rate=crash_count / max(1, num_episodes),
        reward_mean=r_mean,
        reward_std=r_std,
        reward_skew=r_skew,
        reward_kurtosis=r_kurt,
        next_state_entropy=entropy,
        branching_factor=branching_factor,
        observability_gap=observability_gap,
        transition_stochasticity=transition_stochasticity,
        reward_sparsity=reward_sparsity,
        mean_reward_gap=mean_reward_gap,
        mean_first_reward_step=mean_first_reward_step,
    )


# ---------------------------------------------------------------------------
# Probe learning curves


class ReturnLogger(BaseCallback):
    def __init__(self, eval_env: gym.Env, eval_every: int = 5_000, horizon: int = 5, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_every = eval_every
        self.horizon = horizon
        self.timesteps: List[int] = []
        self.returns: List[float] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0 and self.num_timesteps > 0:
            scores: List[float] = []
            for _ in range(self.horizon):
                obs, _ = self.eval_env.reset()
                # obs = _flatten_obs(obs)
                done = False
                total = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    next_obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    # obs = _flatten_obs(next_obs)
                    total += float(reward)
                    done = terminated or truncated
                    obs = next_obs
                scores.append(total)
            self.timesteps.append(int(self.num_timesteps))
            self.returns.append(float(np.mean(scores)))
        return True


# ---------------------------------------------------------------------------
# Portfolio evaluation


PortfolioResult = Dict[str, float]


def make_portfolio(device: str = "cpu") -> Dict[str, Callable[[gym.Env], Any]]:
    policy_kwargs = dict(net_arch=[64, 64])
    return {
        # "DQN": lambda env: DQN("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device),
        # "PPO": lambda env: PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device),
        "A2C": lambda env: A2C("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device),
        # "TD3": lambda env: TD3("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device=device),
    }


def evaluate_algorithm(
    algo_name: str,
    algo_factory: Callable[[gym.Env], Any],
    make_env: EnvMaker,
    steps: int,
    seed: int,
    eval_episodes: int = 10,
) -> float:
    set_global_seed(seed)
    env = make_env()
    eval_env = make_env()

    model = algo_factory(env)
    model.set_random_seed(seed)

    model.learn(total_timesteps=steps)

    returns: List[float] = []
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        # obs = _flatten_obs(obs)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = eval_env.step(action)
            # obs = _flatten_obs(next_obs)
            total += float(reward)
            done = terminated or truncated
            obs = next_obs
        returns.append(total)

    env.close()
    eval_env.close()
    return float(np.mean(returns))


def evaluate_portfolio(
    cfg: EnvConfig,
    steps: int = 200_000,
    seeds: Sequence[int] = (0, 1, 2),
) -> Tuple[PortfolioResult, str]:
    results: PortfolioResult = {}
    portfolio = make_portfolio()

    for algo_name, builder in portfolio.items():
        algo_returns = []
        for seed in seeds:
            start = time.time()
            score = evaluate_algorithm(algo_name, builder, cfg.make_fn, steps, seed)
            elapsed = time.time() - start
            print(f"[portfolio] {cfg.env_id} {algo_name} seed={seed} return={score:.2f} elapsed={elapsed/60:.2f} min")
            algo_returns.append(score)
        results[algo_name] = float(np.mean(algo_returns))

    best_algo = max(results, key=results.get) # type: ignore
    return results, best_algo


# ---------------------------------------------------------------------------
# Meta-feature assembly


def compute_meta_for_instance(
    cfg: EnvConfig,
    probe_steps: int = 50_000,
    portfolio_steps: int = 200_000,
    probe_seeds: Sequence[int] = (0, 1, 2),
    portfolio_seeds: Sequence[int] = (0, 1, 2),
) -> Tuple[MetaFeatures, PortfolioResult, str]:

    env = cfg.make_fn()
    try:
        structural = estimate_structural(env)
    finally:
        env.close()

    portfolio_result, best_algo = evaluate_portfolio(cfg, steps=portfolio_steps, seeds=portfolio_seeds)
    print(f"Finished evaluate portfolio: {portfolio_result}, {best_algo}")

    meta = MetaFeatures(
        env_id=cfg.env_id,
        structural=structural,
        domain_params=cfg.params,
    )
    return meta, portfolio_result, best_algo


def build_feature_matrix(metas: Iterable[MetaFeatures]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for meta in metas:
        record: Dict[str, Any] = {"env_id": meta.env_id}
        record.update({f"param_{k}": v for k, v in meta.domain_params.items()})
        if meta.structural is not None:
            record.update({f"struct_{k}": v for k, v in asdict(meta.structural).items()})
        probe = getattr(meta, "probe", None)
        if probe is not None:
            record.update({f"probe_{k}": v for k, v in asdict(probe).items()})
        traj = getattr(meta, "traj", None)
        if traj is not None and hasattr(traj, "traj_embed_mean"):
            for i, val in enumerate(traj.traj_embed_mean):
                record[f"traj_{i:02d}"] = float(val)
        records.append(record)
    return pd.DataFrame(records)


def build_env_configurations(
    count: int,
    seed: int = 0,
) -> List[EnvConfig]:
    rng = np.random.default_rng(seed)
    configs: List[EnvConfig] = []
    param_grid = []

    traffic_levels = np.linspace(10, 100, num=10)
    speeds = [(18, 25), (20, 30), (25, 35), (30, 40)]
    lane_counts = [3, 4, 5]
    durations = [30, 40, 60]
    

    for veh_cnt in traffic_levels:
        for speed_range in speeds:
            for lanes in lane_counts:
                param_grid.append(
                    {
                        "vehicles_count": int(veh_cnt),
                        "reward_speed_range": list(speed_range),
                        "lanes_count": lanes,
                        "duration": rng.choice(durations),
                    }
                )

    rng.shuffle(param_grid)
    selected_configs = param_grid[:count]

    for idx, params in enumerate(selected_configs):
        env_id = f"highway_{idx:03d}"

        def make_env_closure(config_params: Dict[str, Any]) -> EnvMaker:
            def _make() -> gym.Env:
                base = {
                    "observation": {
                        "type": "Kinematics",
                    },
                    "action": {
                        "type": "DiscreteMetaAction",
                    },
                }
                config = {**config_params, **base}
                env = gym.make("highway-fast-v0", render_mode=None, config=config)
                return env

            return _make

        configs.append(
            EnvConfig(
                env_id=env_id,
                make_fn=make_env_closure(params),
                params=params,
            )
        )

    return configs


# ---------------------------------------------------------------------------
# CLI orchestration


def run_pipeline(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.output_dir))
    configs = build_env_configurations(args.instances, seed=args.seed)

    metas: List[MetaFeatures] = []
    portfolio_returns: Dict[str, Dict[str, float]] = {}
    winners: Dict[str, str] = {}

    for cfg in configs:
        print(f"[meta] Processing {cfg.env_id} with params={json.dumps(cfg.params, default=_json_default)}")
        meta, portfolio_result, best_algo = compute_meta_for_instance(
            cfg,
            probe_steps=args.probe_steps,
            portfolio_steps=args.portfolio_steps,
            probe_seeds=tuple(args.probe_seeds),
            portfolio_seeds=tuple(args.portfolio_seeds),
        )
        metas.append(meta)
        portfolio_returns[cfg.env_id] = portfolio_result
        winners[cfg.env_id] = best_algo

    df_features = build_feature_matrix(metas)
    df_returns = pd.DataFrame.from_dict(portfolio_returns, orient="index")
    df_returns.insert(0, "env_id", df_returns.index)
    df_returns.reset_index(drop=True, inplace=True)
    df_winners = pd.DataFrame(
        [{"env_id": env_id, "best_algo": algo} for env_id, algo in winners.items()]
    )

    feature_path = Path(args.output_dir) / "highway_meta_features.parquet"
    returns_path = Path(args.output_dir) / "highway_portfolio_returns.parquet"
    winners_path = Path(args.output_dir) / "highway_winners.parquet"

    df_features.to_parquet(feature_path)
    df_returns.to_parquet(returns_path)
    df_winners.to_parquet(winners_path)

    print(f"Saved features -> {feature_path}")
    print(f"Saved portfolio returns -> {returns_path}")
    print(f"Saved winners -> {winners_path}")

DEFAULT_ENV_CONFIG = Path("config") / "env-configurations.yaml"
DEFAULT_ALGO_CONFIG = Path("config") / "algo-configurations.yaml"
DEFAULT_ALGO_PRESETS = Path("config") / "algo-presets.yaml"

PRESET_WEIGHTS: Dict[str, float] = {
    "conservative": 0.2,
    "balanced": 0.5,
    "aggressive": 0.8,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instance space analysis for highway-v0.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for artifacts.")
    parser.add_argument("--instances", type=int, default=24, help="Number of environment instances.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for configuration sampling.")
    parser.add_argument("--probe-steps", type=int, default=50_000, help="Training steps for probe agents.")
    parser.add_argument("--portfolio-steps", type=int, default=200_000, help="Training steps for portfolio agents.")
    parser.add_argument(
        "--probe-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds for probe runs.",
    )
    parser.add_argument(
        "--portfolio-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds for portfolio runs.",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=str(DEFAULT_ENV_CONFIG),
        required=False,
        help="Path to the yaml environment configurations file.",
    )
    parser.add_argument(
        "--algo-config",
        type=str,
        default=str(DEFAULT_ALGO_CONFIG),
        required=False,
        help="Path to the yaml algorithm configuration ranges file.",
    )
    parser.add_argument(
        "--algo-presets",
        type=str,
        default=str(DEFAULT_ALGO_PRESETS),
        required=False,
        help="Path where the generated algorithm presets should be written.",
    )
    parser.add_argument(
        "--only-generate-presets",
        action="store_true",
        help="Generate algorithm presets and exit without running the pipeline.",
    )
    return parser.parse_args(argv)


def generate_algo_configs(
    config_path: Path,
    output_path: Path,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if weights is None:
        weights = PRESET_WEIGHTS

    with config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle) or {}

    presets: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for algo_name, params in base_config.items():
        algo_presets: Dict[str, Dict[str, Any]] = {}
        for preset_name, weight in weights.items():
            preset_values: Dict[str, Any] = {}
            for param_name, bounds in params.items():
                if isinstance(bounds, list) and len(bounds) == 2:
                    numeric_bounds = [_coerce_numeric(bound) for bound in bounds]
                    if all(val is not None for val in numeric_bounds):
                        lo_val, hi_val = numeric_bounds  # type: ignore[misc]
                        if float(lo_val).is_integer() and float(hi_val).is_integer():
                            preset_values[param_name] = _interpolate_range_value(int(lo_val), int(hi_val), weight)
                        else:
                            preset_values[param_name] = _interpolate_range_value(float(lo_val), float(hi_val), weight)
                        continue
                preset_values[param_name] = bounds
            algo_presets[preset_name] = preset_values
        presets[algo_name] = algo_presets

    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(presets, handle, sort_keys=False)

    print(f"Saved algorithm presets -> {output_path}")
    return presets


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    algo_config_path = Path(args.algo_config)
    algo_presets_path = Path(args.algo_presets)
    generate_algo_configs(algo_config_path, algo_presets_path)
    if args.only_generate_presets:
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
