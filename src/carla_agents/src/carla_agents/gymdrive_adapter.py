import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register


REPO_ROOT = Path(__file__).resolve().parents[4]
CARLA_GYMDRIVE_ROOT = REPO_ROOT / "CARLA-GymDrive"
ENV_ID = "isa-carla-gymdrive-v0"


def register_carla_env() -> None:
    try:
        gym.spec(ENV_ID)
        return
    except gym.error.Error:
        pass

    register(
        id=ENV_ID,
        entry_point="carla_agents.gymdrive_adapter:IsaCarlaGymDriveEnv",
    )


def ensure_carla_gymdrive_on_path() -> None:
    carla_path = str(CARLA_GYMDRIVE_ROOT)
    if carla_path not in sys.path:
        sys.path.insert(0, carla_path)


def resolve_carla_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path

    carla_path = CARLA_GYMDRIVE_ROOT / path
    if carla_path.exists():
        return carla_path

    raise FileNotFoundError(f"CARLA path does not exist: {value}")


class IsaCarlaGymDriveEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None):
        super().__init__()
        ensure_carla_gymdrive_on_path()

        self.config = dict(config)
        self.orig_config = dict(config)
        self.render_mode = render_mode
        self._autopilot_enabled = bool(self.config["autopilot"])

        scenarios_file = resolve_carla_path(self.config["scenarios_file"])
        sensor_config = resolve_carla_path(self.config["sensor_config"])
        os.environ["CARLA_GYMDRIVE_SCENARIOS_FILE"] = os.fspath(scenarios_file)
        os.environ["CARLA_GYMDRIVE_SENSORS_FILE"] = os.fspath(sensor_config)

        from src.env.environment import CarlaEnv  # noqa: F401
        from src.env.rl_observation_wrapper import RlObservationWrapper
        from src.env.vae_observation_wrapper import VaeObservationWrapper

        if self.config["initialize_server"]:
            raise ValueError("CARLA ISA integration expects initialize_server to be false.")

        env = gym.make(
            "carla-rl-gym-v0",
            max_episode_steps=self.config["max_steps"],
            initialize_server=self.config["initialize_server"],
            random_weather=self.config["random_weather"],
            random_traffic=self.config["random_traffic"],
            synchronous_mode=self.config["synchronous_mode"],
            continuous=self.config["continuous_actions"],
            show_sensor_data=self.config["show_sensor_data"],
            has_traffic=self.config["has_traffic"],
            autopilot=self._autopilot_enabled,
            verbose=self.config["verbose"],
            scenario_names=list(self.config["scenario_names"]),
        )

        if self.config["use_vae"]:
            env = VaeObservationWrapper(
                env,
                model_name=self.config["vae_model"],
                vae_root=resolve_carla_path(self.config["vae_root"]),
                device=self.config["vae_device"],
                keep_rgb=self.config["keep_rgb"],
                use_mean=not self.config["sample_latent"],
            )
        env = RlObservationWrapper(env)

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @property
    def unwrapped(self) -> gym.Env:
        return self

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def enable_autopilot(self, enabled: bool) -> None:
        self._autopilot_enabled = enabled
        self._set_private_attr("__autopilot", enabled)

    def set_autopilot_route_commands(self, route_commands: Optional[List[str]]) -> None:
        self._set_private_attr("__autopilot_route_commands", route_commands)

    def get_current_speed(self) -> float:
        vehicle = self._get_private_attr("__vehicle")
        return float(vehicle.get_speed())

    def get_current_position(self) -> np.ndarray:
        vehicle = self._get_private_attr("__vehicle")
        location = vehicle.get_location()
        return np.asarray([location.x, location.y, location.z], dtype=np.float32)

    def get_active_scenario(self) -> Dict[str, Any]:
        scenario = self._get_private_attr("__active_scenario_dict")
        return dict(scenario)

    def get_active_scenario_name(self) -> str:
        return str(self._get_private_attr("__active_scenario_name"))

    def _base_env(self) -> Any:
        current = self.env
        while hasattr(current, "env"):
            current = current.env
        return current.unwrapped

    def _get_private_attr(self, name: str) -> Any:
        base_env = self._base_env()
        attr_name = f"_{base_env.__class__.__name__}{name}"
        return getattr(base_env, attr_name)

    def _set_private_attr(self, name: str, value: Any) -> None:
        base_env = self._base_env()
        attr_name = f"_{base_env.__class__.__name__}{name}"
        setattr(base_env, attr_name, value)


register_carla_env()
