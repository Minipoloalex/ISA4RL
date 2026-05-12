import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return mean(values)


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def load_probe_features(instance_path: Path) -> Optional[Dict[str, float]]:
    metafeatures_path = instance_path / "metafeatures.json"
    if not metafeatures_path.is_file():
        return None

    metafeatures = read_json(metafeatures_path)
    feature_groups = metafeatures["feature_groups"]
    if "probes" not in feature_groups:
        return None

    return feature_groups["probes"]["features"]


def summarize_eval(eval_path: Path) -> Dict[str, Any]:
    episodes = read_json(eval_path)
    rewards: List[float] = []
    lengths: List[float] = []
    speeds: List[float] = []
    crashed: List[float] = []
    success: List[float] = []
    first_actions: Counter[str] = Counter()
    all_actions: Counter[str] = Counter()
    high_speed_rewards: List[float] = []
    goal_rewards: List[float] = []
    collision_rewards: List[float] = []

    for episode in episodes:
        rewards.append(float(episode["reward"]))
        lengths.append(float(episode["length"]))

        infos = episode["infos"]
        episode_crashed = False
        episode_success = False
        episode_first_action_recorded = False
        for info in infos:
            if "speed" in info:
                speeds.append(float(info["speed"]))
            if "crashed" in info and bool(info["crashed"]):
                episode_crashed = True
            if "is_success" in info and bool(info["is_success"]):
                episode_success = True
            if "success" in info and bool(info["success"]):
                episode_success = True
            if "action" in info:
                action = str(info["action"])
                all_actions[action] += 1
                if not episode_first_action_recorded:
                    first_actions[action] += 1
                    episode_first_action_recorded = True
            if "rewards" in info:
                reward_parts = info["rewards"]
                if "high_speed_reward" in reward_parts:
                    high_speed_rewards.append(float(reward_parts["high_speed_reward"]))
                if "goal_reward" in reward_parts:
                    goal_rewards.append(float(reward_parts["goal_reward"]))
                if "collision_reward" in reward_parts:
                    collision_rewards.append(float(reward_parts["collision_reward"]))

        crashed.append(float(episode_crashed))
        success.append(float(episode_success))

    return {
        "episodes": len(episodes),
        "mean_reward": safe_mean(rewards),
        "mean_length": safe_mean(lengths),
        "mean_speed": safe_mean(speeds),
        "crash_rate": safe_mean(crashed),
        "success_rate": safe_mean(success),
        "first_actions": first_actions,
        "all_actions": all_actions,
        "mean_high_speed_reward_component": safe_mean(high_speed_rewards),
        "mean_goal_reward_component": safe_mean(goal_rewards),
        "mean_collision_reward_component": safe_mean(collision_rewards),
    }


def summarize_eval_callback(eval_callback_path: Path) -> Dict[str, Optional[float]]:
    if not eval_callback_path.is_file():
        return {
            "eval_callback_best_mean_reward": None,
            "eval_callback_last_mean_reward": None,
            "eval_callback_best_success_rate": None,
        }

    data = np.load(eval_callback_path)
    results = data["results"]
    if results.size == 0:
        return {
            "eval_callback_best_mean_reward": None,
            "eval_callback_last_mean_reward": None,
            "eval_callback_best_success_rate": None,
        }

    mean_rewards = results.mean(axis=1)
    best_index = int(mean_rewards.argmax())
    best_success_rate = None
    if "successes" in data.files:
        best_success_rate = float(np.asarray(data["successes"][best_index]).mean())

    return {
        "eval_callback_best_mean_reward": float(mean_rewards.max()),
        "eval_callback_last_mean_reward": float(mean_rewards[-1]),
        "eval_callback_best_success_rate": best_success_rate,
    }


def load_algo_name(train_run_path: Path) -> str:
    algo_config_path = train_run_path / "algo_config.json"
    if not algo_config_path.is_file():
        return train_run_path.name

    algo_config = read_json(algo_config_path)
    if "algo" not in algo_config:
        return train_run_path.name
    return str(algo_config["algo"]).lower()


def iter_instance_paths(results_path: Path, env_name: Optional[str]) -> Iterable[Path]:
    env_paths = [results_path / env_name] if env_name is not None else sorted(results_path.iterdir())
    for env_path in env_paths:
        if not env_path.is_dir() or env_path.name == "isa":
            continue
        for instance_path in sorted(env_path.iterdir()):
            if instance_path.is_dir() and (instance_path / "metafeatures.json").is_file():
                yield instance_path


def action_distribution_text(counter: Counter[str], max_items: int = 5) -> str:
    total = sum(counter.values())
    if total == 0:
        return "n/a"
    parts = []
    for action, count in counter.most_common(max_items):
        parts.append(f"{action}:{count / total:.2f}")
    return ", ".join(parts)


def print_run_detail(record: Dict[str, Any]) -> None:
    print(
        f"{record['env']}/{record['instance']} {record['algo']} "
        f"delta={record['delta']:.4f}"
    )
    print(
        "  reward: "
        f"agent={format_float(record['agent_mean_reward'])}, "
        f"random={format_float(record['random_mean_reward'])}, "
        f"baseline={format_float(record['baseline_mean_reward'])}"
    )
    print(
        "  agent behavior: "
        f"length={format_float(record['agent_mean_length'])}, "
        f"speed={format_float(record['agent_mean_speed'])}, "
        f"crash_rate={format_float(record['agent_crash_rate'])}, "
        f"success_rate={format_float(record['agent_success_rate'])}"
    )
    print(
        "  agent reward parts: "
        f"high_speed={format_float(record['agent_mean_high_speed_reward_component'])}, "
        f"goal={format_float(record['agent_mean_goal_reward_component'])}, "
        f"collision={format_float(record['agent_mean_collision_reward_component'])}"
    )
    print(
        "  training eval callback: "
        f"best={format_float(record['eval_callback_best_mean_reward'])}, "
        f"last={format_float(record['eval_callback_last_mean_reward'])}, "
        f"best_success_rate={format_float(record['eval_callback_best_success_rate'])}, "
        f"json_vs_best_delta={format_float(record['json_vs_eval_callback_best_delta'])}"
    )
    print(
        "  random probe: "
        f"collision_rate={format_float(record['random_collision_rate'])}, "
        f"timeout_rate={format_float(record['random_timeout_rate'])}, "
        f"speed_snr={format_float(record['random_speed_snr'])}"
    )
    print(f"  first actions: {action_distribution_text(record['first_actions'])}")
    print(f"  all actions:   {action_distribution_text(record['all_actions'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stored random probe reward against stored trained-agent "
            "evaluations and print diagnostics for runs below random."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Results folder to inspect. Defaults to the project results folder.",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Optional environment folder name to inspect, for example 'exit'.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of worst agent-minus-random runs to print.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on instances to inspect for a fast sample.",
    )
    parser.add_argument(
        "--eval-callback-gap-delta",
        type=float,
        default=1.0,
        help=(
            "Report runs where eval_results.json mean reward is this much lower "
            "than the best training-time mean reward stored in evaluations.npz. "
            "This is an optimism/noise diagnostic, not necessarily an artifact bug."
        ),
    )
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    env_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"runs": 0, "below": 0})
    inspected_instances = 0

    for instance_path in iter_instance_paths(args.results, args.env):
        if args.max_instances is not None and inspected_instances >= args.max_instances:
            break
        inspected_instances += 1

        probe_features = load_probe_features(instance_path)
        if probe_features is None or "random_mean_reward" not in probe_features:
            continue

        random_mean_reward = float(probe_features["random_mean_reward"])
        baseline_mean_reward = (
            float(probe_features["baseline_mean_reward"])
            if "baseline_mean_reward" in probe_features
            else None
        )
        train_path = instance_path / "train"
        if not train_path.is_dir():
            continue

        for train_run_path in sorted(train_path.iterdir()):
            eval_path = train_run_path / "eval_results.json"
            if not eval_path.is_file():
                continue

            eval_summary = summarize_eval(eval_path)
            eval_callback_summary = summarize_eval_callback(
                train_run_path / "evaluations.npz"
            )
            agent_mean_reward = eval_summary["mean_reward"]
            if agent_mean_reward is None:
                continue

            env_name = instance_path.parent.name
            delta = float(agent_mean_reward) - random_mean_reward
            env_counts[env_name]["runs"] += 1
            if delta < 0.0:
                env_counts[env_name]["below"] += 1
            eval_callback_best = eval_callback_summary[
                "eval_callback_best_mean_reward"
            ]
            json_vs_eval_callback_best_delta = (
                float(agent_mean_reward) - eval_callback_best
                if eval_callback_best is not None
                else None
            )

            records.append(
                {
                    "env": env_name,
                    "instance": instance_path.name,
                    "algo": load_algo_name(train_run_path),
                    "delta": delta,
                    "random_mean_reward": random_mean_reward,
                    "baseline_mean_reward": baseline_mean_reward,
                    "random_collision_rate": probe_features["random_collision_rate"]
                    if "random_collision_rate" in probe_features
                    else None,
                    "random_timeout_rate": probe_features["random_timeout_rate"]
                    if "random_timeout_rate" in probe_features
                    else None,
                    "random_speed_snr": probe_features["random_speed_snr"]
                    if "random_speed_snr" in probe_features
                    else None,
                    "agent_mean_reward": agent_mean_reward,
                    "agent_mean_length": eval_summary["mean_length"],
                    "agent_mean_speed": eval_summary["mean_speed"],
                    "agent_crash_rate": eval_summary["crash_rate"],
                    "agent_success_rate": eval_summary["success_rate"],
                    "agent_mean_high_speed_reward_component": eval_summary[
                        "mean_high_speed_reward_component"
                    ],
                    "agent_mean_goal_reward_component": eval_summary[
                        "mean_goal_reward_component"
                    ],
                    "agent_mean_collision_reward_component": eval_summary[
                        "mean_collision_reward_component"
                    ],
                    "eval_callback_best_mean_reward": eval_callback_best,
                    "eval_callback_last_mean_reward": eval_callback_summary[
                        "eval_callback_last_mean_reward"
                    ],
                    "eval_callback_best_success_rate": eval_callback_summary[
                        "eval_callback_best_success_rate"
                    ],
                    "json_vs_eval_callback_best_delta": json_vs_eval_callback_best_delta,
                    "first_actions": eval_summary["first_actions"],
                    "all_actions": eval_summary["all_actions"],
                }
            )

    total_runs = len(records)
    below_runs = sum(1 for record in records if record["delta"] < 0.0)
    print(f"Inspected instances: {inspected_instances}")
    print(f"Agent runs with random probe comparison: {total_runs}")
    print(f"Runs below random: {below_runs}")
    if total_runs:
        print(f"Share below random: {below_runs / total_runs:.3f}")

    suspicious_records = [
        record
        for record in records
        if record["json_vs_eval_callback_best_delta"] is not None
        and record["json_vs_eval_callback_best_delta"] < -args.eval_callback_gap_delta
    ]
    print(
        "Runs with training-callback optimism gap: "
        f"{len(suspicious_records)}"
    )

    print("\nBy environment:")
    for env_name in sorted(env_counts):
        runs = env_counts[env_name]["runs"]
        below = env_counts[env_name]["below"]
        share = below / runs if runs else 0.0
        print(f"  {env_name}: {below}/{runs} below random ({share:.3f})")

    below_records = sorted(
        [record for record in records if record["delta"] < 0.0],
        key=lambda record: record["delta"],
    )
    if not below_records:
        return

    if suspicious_records:
        print(
            f"\nWorst {min(args.top, len(suspicious_records))} "
            "training-callback optimism gaps:"
        )
        for record in sorted(
            suspicious_records,
            key=lambda item: item["json_vs_eval_callback_best_delta"],
        )[: args.top]:
            print_run_detail(record)

    print(f"\nWorst {min(args.top, len(below_records))} runs below random:")
    for record in below_records[: args.top]:
        print_run_detail(record)


if __name__ == "__main__":
    main()
