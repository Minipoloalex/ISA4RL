import argparse
import time

import gymnasium as gym
import numpy as np
from panda3d.core import loadPrcFileData
from stable_baselines3 import PPO

from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.my_metadrive_env import MyMetaDriveEnv
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.observation_base import BaseObservation
import cupy as cp

loadPrcFileData("", "notify-level-linmath error")


class Sb3MetaDriveObservationWrapper(gym.ObservationWrapper):
    """Convert MetaDrive image observations into SB3-friendly NumPy tensors."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self._convert_space(env.observation_space)

    def observation(self, observation):
        return self._convert_observation(observation)

    def _convert_space(self, space):
        if isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict(
                {key: self._convert_space(subspace) for key, subspace in space.spaces.items()}
            )

        if isinstance(space, gym.spaces.Box) and len(space.shape) == 4:
            height, width, channels, frames = space.shape
            return gym.spaces.Box(
                low=float(np.min(space.low)),
                high=float(np.max(space.high)),
                shape=(channels * frames, height, width),
                dtype=space.dtype,
            )

        if isinstance(space, gym.spaces.Box) and len(space.shape) == 3:
            height, width, channels = space.shape
            return gym.spaces.Box(
                low=float(np.min(space.low)),
                high=float(np.max(space.high)),
                shape=(channels, height, width),
                dtype=space.dtype,
            )

        return space

    def _convert_observation(self, observation):
        if cp is not None and isinstance(observation, cp.ndarray):
            observation = cp.asnumpy(observation)

        if isinstance(observation, dict):
            return {key: self._convert_observation(value) for key, value in observation.items()}

        if isinstance(observation, np.ndarray) and observation.ndim == 4:
            height, width, channels, frames = observation.shape
            observation = observation.reshape(height, width, channels * frames)
            return np.transpose(observation, (2, 0, 1))

        if isinstance(observation, np.ndarray) and observation.ndim == 3:
            return np.transpose(observation, (2, 0, 1))

        return observation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "play", "train_play"],
        default="train",
        help="Train a model, play a saved model with rendering, or do both in sequence.",
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--image-on-cuda", action="store_true")
    parser.add_argument("--model-path", default="metadrive_image_state_ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def build_env_config(args, use_render):
    return {
        "use_render": use_render,
        "random_traffic": False,
        "num_scenarios": 1000000000000,
        "traffic_mode": "respawn",
        "accident_prob": 0,
        "random_agent_model": False,
        "store_map": False,
        "log_level": 20,
        "traffic_density": 0.0,
        "discrete_action": True,
        "horizon": 800,
        "image_observation": True,
        "map_config": {
            "type": "block_sequence",
            "config": "rORY",
            "lane_width": 3.0,
            "lane_num": 2
        },
        "vehicle_config": {
            "vehicle_model": "default"
        },
    }

def make_env(args, use_render):
    env = MyMetaDriveEnv(config=build_env_config(args, use_render=use_render))
    return Sb3MetaDriveObservationWrapper(env)


def build_model(args, env):
    policy = "MultiInputPolicy"
    return PPO(
        policy,
        env,
        verbose=1,
        policy_kwargs=dict(normalize_images=False),
    )


def train(args):
    env = make_env(args, use_render=False)
    model = build_model(args, env)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_path)
    env.close()


def play(args):
    env = make_env(args, use_render=True)
    model = PPO.load(args.model_path, env=env)

    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            terminated = False
            truncated = False
            step = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                env.render()
                if args.sleep > 0:
                    time.sleep(args.sleep)

            print(
                f"Episode {episode + 1}/{args.episodes} finished after {step} steps "
                f"with reward {episode_reward:.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    args = parse_args()

    if args.mode in {"train", "train_play"}:
        train(args)

    if args.mode in {"play", "train_play"}:
        play(args)
