from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, SupportsFloat
from copy import deepcopy
from types import MethodType

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
        return 1
    elif isinstance(env.action_space, gym.spaces.Box):
        return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
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


def _build_idm_vehicle(vehicle: Any) -> Any:
    if isinstance(vehicle, IDMVehicle):
        return vehicle

    if hasattr(vehicle, "target_lane_index"):
        idm_vehicle = IDMVehicle.create_from(vehicle)
    else:
        lane_index = getattr(vehicle, "lane_index", None)
        route = getattr(vehicle, "route", None)
        if route is None and lane_index is not None:
            route = [lane_index]

        idm_vehicle = IDMVehicle(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=lane_index,
            target_speed=getattr(vehicle, "target_speed", vehicle.speed),
            route=route,
            timer=getattr(vehicle, "timer", None),
        )

    for attr in (
        "color",
        "action",
        "crashed",
        "impact",
        "log",
        "history",
        "prediction_type",
    ):
        _copy_vehicle_attr(vehicle, idm_vehicle, attr, deep_copy=attr == "impact")

    return idm_vehicle

def _route_to_exit_lane(base_env: gym.Env) -> Optional[List[Tuple[str, str, int]]]:
    config = getattr(base_env, "config", {})
    lane_count = config["lanes_count"]
    network = base_env.road.network
    required_edges = [("0", "1"), ("1", "2"), ("2", "exit")]
    if any(_from not in network.graph or _to not in network.graph[_from] for _from, _to in required_edges):
        return None

    mainline_rightmost_lane = lane_count - 1
    exit_entry_lane = lane_count
    return [
        ("0", "1", mainline_rightmost_lane),
        ("1", "2", exit_entry_lane),
        ("2", "exit", 0),
    ]

def _configure_idm_route(base_env: gym.Env, idm_vehicle: Any) -> None:
    if getattr(idm_vehicle, "route", None):
        return

    env_id = base_env.spec.id
    if env_id != "exit-v0":
        return

    exit_route = _route_to_exit_lane(base_env)
    if exit_route is not None:
        idm_vehicle.route = exit_route

def _distance_to_exit_decision_point(base_env: gym.Env, idm_vehicle: Any) -> Optional[float]:
    lane_index = getattr(idm_vehicle, "lane_index", None)
    lane = getattr(idm_vehicle, "lane", None)
    config = getattr(base_env, "config", {})
    if lane_index is None or lane is None:
        return None

    longitudinal, _ = lane.local_coordinates(idm_vehicle.position)
    road_key = lane_index[:2]
    if road_key == ("0", "1"):
        return float(config["exit_position"] - longitudinal)
    if road_key == ("1", "2"):
        return float(config["exit_length"] - longitudinal)
    if road_key == ("2", "exit"):
        return float("inf")
    return None

def _desired_exit_lane(base_env: gym.Env, lane_index: Tuple[str, str, int]) -> Optional[int]:
    lane_count = getattr(base_env, "config", {})["lanes_count"]
    road_key = lane_index[:2]
    if road_key == ("0", "1"):
        return lane_count - 1
    if road_key == ("1", "2"):
        return lane_count
    if road_key == ("2", "exit"):
        return 0
    return None

def _configure_exit_idm_behavior(base_env: gym.Env, idm_vehicle: Any) -> None:
    env_id = base_env.spec.id
    if env_id != "exit-v0":
        return

    idm_vehicle.POLITENESS = -1
    idm_vehicle.LANE_CHANGE_MIN_ACC_GAIN = -0.2
    idm_vehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED = 6
    idm_vehicle.TIME_WANTED = 0.5
    idm_vehicle.COMFORT_ACC_MAX = 4.0
    idm_vehicle.COMFORT_ACC_MIN = -6.0


def ensure_idm_vehicle(env: gym.Env) -> None:
    base_env = env.unwrapped
    vehicle = base_env.vehicle
    road = base_env.road

    idm_vehicle = _build_idm_vehicle(vehicle)
    idm_vehicle.enable_lane_change = True
    idm_vehicle.timer = getattr(vehicle, "timer", getattr(idm_vehicle, "timer", 0.0))
    _copy_vehicle_attr(vehicle, idm_vehicle, "target_speed")
    _copy_vehicle_attr(vehicle, idm_vehicle, "target_speeds", deep_copy=True)
    _copy_vehicle_attr(vehicle, idm_vehicle, "index_to_speed")
    _copy_vehicle_attr(vehicle, idm_vehicle, "speed_index")

    # both just for exit environment (planning route + changing behavior)
    _configure_idm_route(base_env, idm_vehicle)
    _configure_exit_idm_behavior(base_env, idm_vehicle)

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

def get_max_episode_steps(env: gym.Env) -> int:
    config = env.unwrapped.config
    try:
        return config["duration"] * config["policy_frequency"]
    except:
        pass
    try:
        return config["horizon"]
    except:
        pass
    raise NotImplementedError(f"Max episode steps not implemented for env: {env}")

def _env_viewer_deepcopy(self, memo):
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result

    new_env = deepcopy(self.env, memo)
    new_config = deepcopy(self.config, memo)
    new_config["offscreen_rendering"] = True
    
    result.__init__(new_env, config=new_config)
    return result

def safe_copy_env(env: gym.Env) -> gym.Env:
    # Need to deepcopy the env, but deepcopy doesn't work with pygame's 
    # WorldSurface (used by EnvViewer). We monkey-patch EnvViewer.__deepcopy__ 
    # temporarily to properly reconstruct the viewer without pickling surfaces.
    try:
        from highway_env.envs.common.graphics import EnvViewer
        original_deepcopy = getattr(EnvViewer, "__deepcopy__", None)
        EnvViewer.__deepcopy__ = _env_viewer_deepcopy
        patch_applied = True
    except ImportError:
        patch_applied = False

    try:
        base_env = deepcopy(env)
    finally:
        if patch_applied:
            if original_deepcopy is None:
                del EnvViewer.__deepcopy__
            else:
                EnvViewer.__deepcopy__ = original_deepcopy

    return base_env

def __visualize_idm_exit():
    import time
    env = gym.make("exit-v0", render_mode="human", config={"duration": 20})
    obs, info = env.reset()

    ensure_idm_vehicle(env)
    vehicle = env.unwrapped.vehicle
    print(f"Initial route: {getattr(vehicle, 'route', None)}")
    print(
        "Initial lane state:",
        f"lane_index={getattr(vehicle, 'lane_index', None)}",
        f"target_lane_index={getattr(vehicle, 'target_lane_index', None)}",
    )

    step = 0
    done = truncated = False
    while not (done or truncated):
        action = default_idle_action(env)
        obs, reward, done, truncated, info = env.step(action)
        vehicle = env.unwrapped.vehicle
        print(
            f"step={step:03d}",
            f"lane_index={getattr(vehicle, 'lane_index', None)}",
            f"target_lane_index={getattr(vehicle, 'target_lane_index', None)}",
            f"speed={getattr(vehicle, 'speed', None):.2f}",
            f"route_head={(getattr(vehicle, 'route', None) or [None])[0]}",
            f"is_success={info.get('is_success')}",
        )
        env.render()
        time.sleep(0.2)
        step += 1

    env.close()

if __name__ == "__main__":
    __visualize_idm_exit()
