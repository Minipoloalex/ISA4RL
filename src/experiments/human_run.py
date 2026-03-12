"""Simple CLI to drive a Highway-env scenario with manual inputs."""

from __future__ import annotations
import argparse
from typing import Dict, Tuple, Union, Mapping, Optional
import gymnasium as gym
from gymnasium import spaces
import highway_env  # noqa: F401 - needed to register the environments
import numpy as np

ActionValue = int
KeyBindings = Dict[str, Tuple[ActionValue, str]]
MaybeBindings = Optional[KeyBindings]
ContinuousAction = np.ndarray

KEY_TEMPLATES: Tuple[Tuple[str, str, str], ...] = (
    ("", "IDLE", "keep lane / idle"),
    ("x", "IDLE", "no-op / idle"),
    ("w", "FASTER", "accelerate"),
    ("s", "SLOWER", "brake"),
    ("a", "LANE_LEFT", "lane left"),
    ("d", "LANE_RIGHT", "lane right"),
)
FALLBACK_KEYS = tuple("1234567890abcdefghijklmnopqrstuvwxyz")


def build_key_bindings(env: gym.Env) -> KeyBindings:
    """Create key bindings that match the env's discrete actions."""
    action_type = getattr(env.unwrapped, "action_type", None)
    key_bindings: KeyBindings = {}
    action_lookup: Dict[str, int] = {}
    if action_type is not None and hasattr(action_type, "actions"):
        actions: Mapping[int, str] = action_type.actions  # type: ignore[assignment]
        action_lookup = {name: idx for idx, name in actions.items()}
    elif hasattr(env.action_space, "n"):
        action_lookup = {f"ACTION_{i}": i for i in range(env.action_space.n)}
    else:
        raise RuntimeError("Unable to infer key bindings for the environment.")

    for key, action_name, desc in KEY_TEMPLATES:
        action_idx = action_lookup.get(action_name)
        if action_idx is not None:
            key_bindings[key] = (action_idx, desc)

    if key_bindings:
        return key_bindings

    for idx, key in zip(range(env.action_space.n), FALLBACK_KEYS):
        key_bindings[key] = (idx, f"action {idx}")

    if not key_bindings:
        raise RuntimeError("Unable to infer key bindings for the environment.")
    return key_bindings


def print_help(key_bindings: KeyBindings) -> None:
    """Display available commands."""
    print("\nControls:")
    for key, (_, desc) in key_bindings.items():
        label = "<enter>" if key == "" else key
        print(f"  {label:>7} -> {desc}")
    print("      r -> reset episode")
    print("      q -> quit\n")


def format_prompt(key_bindings: KeyBindings) -> str:
    keys = "/".join("<enter>" if key == "" else key for key in key_bindings)
    return f"Next action [{keys}, r=reset, q=quit]: "


def get_user_action(
    key_bindings: KeyBindings, prompt: str
) -> Union[ActionValue, str]:
    """Prompt the user for the next action."""
    while True:
        cmd = (
            input(prompt).strip().lower()
        )
        if cmd == "q":
            return "quit"
        if cmd == "r":
            return "reset"
        if cmd in key_bindings:
            return key_bindings[cmd][0]
        print("Unknown command, try again.")


def format_continuous_prompt(space: spaces.Box) -> str:
    shape = "x".join(str(dim) for dim in space.shape)
    return (
        "Next action "
        f"({shape} values, comma/space separated, blank=reuse/zero, r=reset, q=quit): "
    )


def parse_continuous(
    raw: str, space: spaces.Box, last_action: Optional[ContinuousAction]
) -> Union[ContinuousAction, str, None]:
    """Parse user input into a continuous action vector.

    Returns:
        np.ndarray: valid action vector.
        'quit' or 'reset' command strings.
        None if parsing failed and the user should be re-prompted.
    """
    raw = raw.strip().lower()
    if raw == "q":
        return "quit"
    if raw == "r":
        return "reset"
    if raw == "":
        if last_action is not None:
            return last_action
        # Default to zeros within bounds
        return np.zeros_like(space.low, dtype=np.float32)

    # Split on comma or whitespace
    tokens = [t for chunk in raw.split(",") for t in chunk.split()]
    try:
        values = np.array([float(t) for t in tokens], dtype=np.float32)
    except ValueError:
        print("Could not parse numbers, try again.")
        return None

    if values.size != np.prod(space.shape):
        print(f"Expected {space.shape} values, got {values.size}. Try again.")
        return None
    values = values.reshape(space.shape)

    clipped = np.clip(values, space.low, space.high)
    if not np.allclose(values, clipped):
        print("Warning: values clipped to action bounds.")
    return clipped


def get_continuous_action(
    space: spaces.Box, prompt: str, last_action: Optional[ContinuousAction]
) -> Union[ContinuousAction, str]:
    """Prompt the user for the next continuous action."""
    while True:
        raw = input(prompt)
        result = parse_continuous(raw, space, last_action)
        if result is not None:
            return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual control helper for highway-env."
    )
    parser.add_argument("--env", default="highway-v0", help="Environment ID to load.")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = {}
    if args.env.startswith("highway"):
        env_config.update(
            {
                "lanes_count": 5,
                "vehicles_count": 50,
            }
        )
    if args.env.startswith("intersection"):
        env_config.update(
            {
                "action": {
                    "type": "DiscreteMetaAction",
                    "longitudinal": True,
                    "lateral": False,
                    "target_speeds": [0, 4.5, 9],
                },
                "manual_control": True,
            }
        )
    if args.env.startswith("exit"):
        env_config.update(
            {
                "lanes_count": 3,
                "vehicles_count": 10,
                "vehicles_density": 1,
            }
        )
    if args.env.startswith("two-way"):
        env_config.update(
            
        )
    if args.env == "racetrack-oval-v0":
        env_config.update({
            "block_lane": False,
            "force_decision": True,
            "no_lanes": 5,
        })
    env_kwargs: Dict[str, object] = {"render_mode": "human"}
    if env_config:
        env_kwargs["config"] = env_config
    env = gym.make(args.env, **env_kwargs)  # type: ignore[arg-type]
    action_space = env.action_space
    is_continuous = isinstance(action_space, spaces.Box)
    key_bindings: MaybeBindings = None
    if not is_continuous:
        key_bindings = build_key_bindings(env)
        print_help(key_bindings)
        prompt: str = format_prompt(key_bindings)
    else:
        prompt = format_continuous_prompt(action_space)
        print(
            "\nContinuous control mode.",
            f"Action bounds low={action_space.low}, high={action_space.high}",
        )
    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        done = truncated = False
        steps = 0
        episode_return = 0.0
        last_action: Optional[ContinuousAction] = None
        print(f"Episode {episode}/{args.episodes}")
        while not (done or truncated):
            if key_bindings is not None:
                action = get_user_action(key_bindings, prompt)
            else:
                action = get_continuous_action(action_space, prompt, last_action)
            if isinstance(action, str) and action == "quit":
                print("Stopping early.")
                env.close()
                return
            if isinstance(action, str) and action == "reset":
                obs, info = env.reset()
                done = truncated = False
                steps = 0
                episode_return = 0.0
                print("Episode reset.")
                continue
            print(action)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            steps += 1
            episode_return += reward
            if is_continuous:
                last_action = action  # type: ignore[assignment]
        print(f"Episode finished after {steps} steps, return={episode_return:.2f}\n")
    env.close()


if __name__ == "__main__":
    main()
