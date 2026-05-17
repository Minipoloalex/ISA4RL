import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import gymnasium as gym
import metadrive  # noqa: F401 - registers metadrive-v0
from metadrive.policy.idm_policy import IDMPolicy


logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_metadrive_configs() -> List[Dict[str, Any]]:
    config_path = _repo_root() / "config" / "env-configs" / "metadrive-configs.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _neutral_action(env: gym.Env) -> Any:
    return env.action_space.sample()


def _active_policy_name(env: gym.Env) -> str:
    base_env = env.unwrapped
    return base_env.engine.get_policy(base_env.vehicle.name).__class__.__name__


def _run_steps(env: gym.Env, steps: int, label: str, seed: int) -> None:
    base_env = env.unwrapped
    for step in range(steps):
        _, reward, terminated, truncated, info = env.step(_neutral_action(env))
        speed = info["velocity"] if "velocity" in info else base_env.vehicle.speed
        logger.info(
            "%s step=%04d reward=%.3f speed=%.3f terminated=%s truncated=%s",
            label,
            step,
            reward,
            speed,
            terminated,
            truncated,
        )
        if terminated or truncated:
            logger.info("%s episode ended. Resetting same env.", label)
            env.reset(seed=seed)


def run_config_switch_demo(
    config_index: int,
    seed: int,
    pre_switch_steps: int,
    steps: int,
    render: bool,
) -> None:
    configs = _load_metadrive_configs()
    selected = configs[config_index]
    env_id = selected["env_id"]
    env_config = deepcopy(selected["config"])

    env_config["use_render"] = render

    env = gym.make(env_id, config=env_config)
    try:
        base_env = env.unwrapped

        logger.info("Created env object id=%s with selected config index=%d", id(base_env), config_index)
        env.reset(seed=seed)
        initial_policy = _active_policy_name(env)
        logger.info("Policy after first reset from original config: %s", initial_policy)

        _run_steps(env, pre_switch_steps, "before-switch", seed)

        logger.info("Mutating already-initialized env config on object id=%s", id(base_env))
        base_env.config["agent_policy"] = IDMPolicy
        base_env.config["manual_control"] = False
        policy_before_reset = _active_policy_name(env)
        logger.info("Policy immediately after config mutation, before reset: %s", policy_before_reset)

        env.reset(seed=seed)
        switched_policy = _active_policy_name(env)
        logger.info("Policy after resetting same env object id=%s: %s", id(base_env), switched_policy)

        if switched_policy != "IDMPolicy":
            raise RuntimeError(f"Expected IDMPolicy after reset, got {switched_policy}.")

        _run_steps(env, steps, "after-switch", seed)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a MetaDrive config after switching the existing env to IDMPolicy before reset."
    )
    parser.add_argument("--config-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pre-switch-steps", type=int, default=0)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_config_switch_demo(
        config_index=args.config_index,
        seed=args.seed,
        pre_switch_steps=args.pre_switch_steps,
        steps=args.steps,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
