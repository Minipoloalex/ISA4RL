from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, SupportsFloat

import numpy as np
import gymnasium as gym
try:
    from highway_env.vehicle.behavior import IDMVehicle
except:
    pass

from .metafeatures.step_info import StepInfo

PolicyFn = Callable[[Any, Dict[str, Any]], Any]
Trajectory = List[StepInfo]

class MetricHook(Protocol):
    def on_probe_start(self) -> None:
        ...

    def on_episode_start(self) -> None:
        ...

    def on_step(self, context: StepInfo) -> None:
        ...

    def on_episode_end(self) -> None:
        ...

    def finalize(self) -> Dict[str, float]:
        ...

def make_random_policy(env: gym.Env) -> PolicyFn:
    action_space = env.action_space

    def policy(_: Any, __: Dict[str, Any]) -> Any:
        return action_space.sample()

    return policy

def default_idle_action(env: gym.Env) -> int:
    if isinstance(env.action_space, gym.spaces.Discrete):
        return 0
    raise NotImplementedError(f"Idle action not implemented for action space type: {type(env.action_space)}")


def constant_policy(action: Any) -> PolicyFn:
    def policy(_: Any, __: Dict[str, Any]) -> Any:
        return action

    return policy

def _copy_vehicle_attr(
    src: Any, dst: Any, attr: str, *, deep_copy: bool = False
) -> None:
    if not hasattr(src, attr):
        return
    value = getattr(src, attr)
    if deep_copy:
        value = None if value is None else np.array(value, copy=True)
    setattr(dst, attr, value)

def ensure_idm_vehicle(env: gym.Env) -> None:
    base_env = env.unwrapped
    vehicle = base_env.vehicle
    road = base_env.road

    idm_vehicle = IDMVehicle.create_from(vehicle)
    idm_vehicle.enable_lane_change = True
    idm_vehicle.timer = getattr(vehicle, "timer", getattr(idm_vehicle, "timer", 0.0))
    _copy_vehicle_attr(vehicle, idm_vehicle, "target_speed")
    _copy_vehicle_attr(vehicle, idm_vehicle, "target_speeds", deep_copy=True)
    _copy_vehicle_attr(vehicle, idm_vehicle, "index_to_speed")
    _copy_vehicle_attr(vehicle, idm_vehicle, "speed_index")

    base_env.vehicle = idm_vehicle

    for idx, existing in enumerate(road.vehicles):
        if existing is vehicle:
            road.vehicles[idx] = idm_vehicle
            break

    base_env.action_type.controlled_vehicle = idm_vehicle


def get_action_space_size(env: gym.Env) -> int:
    if isinstance(env.action_space, gym.spaces.Discrete):
        return env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        return env.action_space.shape[0]
    else:
        raise ValueError(f"Unknown action space type: {env.action_space}")

def get_max_episode_steps(env: gym.Env, config: dict) -> int:
    try:
        return config["duration"] * config["policy_frequency"]
    except:
        pass
    try:
        return config["horizon"]
    except:
        pass
    raise NotImplementedError(f"Max episode steps not implemented for env: {env}")
