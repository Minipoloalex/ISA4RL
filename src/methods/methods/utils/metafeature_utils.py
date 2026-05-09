from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, SupportsFloat
from copy import deepcopy
from types import MethodType
import math

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


def is_parking_env(env: gym.Env) -> bool:
    return env.spec.id == "parking-v0"


def is_lane_keeping_env(env: gym.Env) -> bool:
    return env.spec.id == "lane-keeping-v0"

def is_basic_racetrack_env(env: gym.Env) -> bool:
    return env.spec.id == "racetrack-v0"

def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _continuous_action_ranges(env: gym.Env) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    action_type = env.unwrapped.action_type
    acceleration_range = action_type.acceleration_range
    steering_range = action_type.steering_range
    return tuple(acceleration_range), tuple(steering_range)


def _normalized_continuous_action(
    acceleration: float,
    steering: float,
    acceleration_range: Tuple[float, float],
    steering_range: Tuple[float, float],
) -> np.ndarray:
    def normalize(value: float, value_range: Tuple[float, float]) -> float:
        lo, hi = value_range
        if abs(hi - lo) < 1e-8:
            return 0.0
        return float(np.clip(2.0 * (value - lo) / (hi - lo) - 1.0, -1.0, 1.0))

    return np.asarray(
        [
            normalize(acceleration, acceleration_range),
            normalize(steering, steering_range),
        ],
        dtype=np.float32,
    )


def make_parking_geometric_policy(env: gym.Env) -> PolicyFn:
    """Creates a non-learned goal-seeking controller for highway-env parking.

    The policy uses simulator state, not trained parameters. It acts as the
    parking counterpart to IDM: task-informed, deterministic, and independent
    from the RL algorithms being compared.
    """

    if not isinstance(env.action_space, gym.spaces.Box):
        raise NotImplementedError("Parking geometric policy expects a continuous action space.")

    acceleration_range, steering_range = _continuous_action_ranges(env)
    max_forward_speed = 2.2
    max_reverse_speed = -1.4
    slow_radius = 8.0
    stop_radius = 0.75
    heading_align_radius = 2.5
    kp_speed = 2.0
    kp_steering = 1.35

    def policy(_: Any, __: Dict[str, Any]) -> np.ndarray:
        base_env = env.unwrapped
        vehicle = getattr(base_env, "vehicle", None)
        goal = getattr(vehicle, "goal", None)
        if vehicle is None or goal is None:
            return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        position = np.asarray(vehicle.position, dtype=float)
        goal_position = np.asarray(goal.position, dtype=float)
        delta = goal_position - position
        distance = float(np.linalg.norm(delta))

        heading = float(vehicle.heading)
        goal_heading = float(getattr(goal, "heading", heading))
        speed = float(getattr(vehicle, "speed", 0.0))

        if distance > 1e-6:
            target_heading = math.atan2(float(delta[1]), float(delta[0]))
        else:
            target_heading = goal_heading

        forward_error = _wrap_to_pi(target_heading - heading)
        reverse_error = _wrap_to_pi(target_heading + math.pi - heading)
        reversing = abs(reverse_error) + 0.2 < abs(forward_error)

        if distance < heading_align_radius:
            steering_error = _wrap_to_pi(goal_heading - heading)
            desired_speed = 0.0 if distance < stop_radius and abs(steering_error) < 0.25 else 0.7
            reversing = False
        elif reversing:
            steering_error = reverse_error
            desired_speed = max_reverse_speed * min(distance / slow_radius, 1.0)
        else:
            steering_error = forward_error
            desired_speed = max_forward_speed * min(distance / slow_radius, 1.0)

        acceleration = np.clip(
            kp_speed * (desired_speed - speed),
            acceleration_range[0],
            acceleration_range[1],
        )
        steering = np.clip(
            kp_steering * steering_error,
            steering_range[0],
            steering_range[1],
        )

        return _normalized_continuous_action(
            float(acceleration),
            float(steering),
            acceleration_range,
            steering_range,
        ).astype(env.action_space.dtype)

    return policy


def make_lane_keeping_observation_policy(env: gym.Env) -> PolicyFn:
    """Creates a non-learned steering controller for highway-env lane keeping.

    The controller only reads the observation passed to the policy. For
    AttributesObservation this means it uses the same noisy state and derivative
    values available to learning agents, and the reference state exposed by the
    environment observation.
    """

    if not isinstance(env.action_space, gym.spaces.Box):
        raise NotImplementedError("Lane-keeping policy expects a continuous action space.")

    action_type = env.unwrapped.action_type
    if action_type.longitudinal:
        raise NotImplementedError("Lane-keeping policy expects lateral-only control.")

    _, steering_range = _continuous_action_ranges(env)
    max_abs_steering = max(abs(steering_range[0]), abs(steering_range[1]))
    if max_abs_steering <= 1e-8:
        raise ValueError(f"Invalid steering range for lane-keeping policy: {steering_range}")

    # Decent values
    kp_lateral = 0.08
    kp_heading = 0.95
    kd_lateral = 0.03
    kd_heading = 0.12

    def policy(obs: Any, __: Dict[str, Any]) -> np.ndarray:
        if not isinstance(obs, dict):
            raise TypeError(f"Lane-keeping policy expects dict observations, got {type(obs)}")

        state = np.asarray(obs["state"], dtype=float).reshape(-1)
        derivative = np.asarray(obs["derivative"], dtype=float).reshape(-1)
        reference_state = np.asarray(obs["reference_state"], dtype=float).reshape(-1)

        if state.size < 2 or derivative.size < 2 or reference_state.size < 2:
            raise ValueError(
                "Lane-keeping observation must contain at least lateral and heading components."
            )

        lateral_error = float(state[0] - reference_state[0])
        heading_error = _wrap_to_pi(float(state[1] - reference_state[1]))
        lateral_rate = float(derivative[0])
        heading_rate = float(derivative[1])

        steering = -(
            kp_lateral * lateral_error
            + kp_heading * heading_error
            + kd_lateral * lateral_rate
            + kd_heading * heading_rate
        )
        normalized_steering = np.clip(steering / max_abs_steering, -1.0, 1.0)
        return np.asarray([normalized_steering], dtype=env.action_space.dtype)

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

    # Decent values based on visual experiments
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
    # WorldSurface (used by EnvViewer). We replace EnvViewer.__deepcopy__ 
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
