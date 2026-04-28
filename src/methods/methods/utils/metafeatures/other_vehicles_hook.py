import numpy as np
from typing import Dict, List, Any

from .base_metric_hook import BaseMetricHook
from .step_info import StepInfo


class OtherVehiclesBehaviorHook(BaseMetricHook):
    """
    Tracks the speed variations of non-ego vehicles during each episode.
    Computes the mean speed variation for each vehicle over its lifetime in the episode,
    and returns aggregated statistics (mean, variance, max) of these variations
    over all vehicles to characterize the difficulty/behavior of the traffic.
    """

    def __init__(self):
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.all_vehicle_mean_variations: List[float] = []
        self.all_vehicle_mean_heading_variations: List[float] = []

    def on_probe_start(self) -> None:
        self._reset_stats()

    def on_episode_start(self) -> None:
        # Maps Python object ID of each vehicle to its history in the current episode
        self.current_episode_vehicle_speeds: Dict[int, List[float]] = {}
        self.current_episode_vehicle_headings: Dict[int, List[float]] = {}

    def on_step(self, context: StepInfo) -> None:
        other_vehicles = context.info.get("other_vehicles_states", [])
        for v_data in other_vehicles:
            v_id = v_data["id"]
            speed = v_data["speed"]
            heading = v_data["heading"]
            
            if v_id not in self.current_episode_vehicle_speeds:
                self.current_episode_vehicle_speeds[v_id] = []
                self.current_episode_vehicle_headings[v_id] = []
            
            self.current_episode_vehicle_speeds[v_id].append(speed)
            self.current_episode_vehicle_headings[v_id].append(heading)

    def on_episode_end(self) -> None:
        # Calculate mean variation for each vehicle in the episode
        for v_id, speeds in self.current_episode_vehicle_speeds.items():
            if len(speeds) > 1:
                variations = np.abs(np.diff(speeds))
                mean_var = float(np.mean(variations))
                self.all_vehicle_mean_variations.append(mean_var)
                
                # Heading variation
                headings = np.array(self.current_episode_vehicle_headings[v_id])
                diffs = headings[1:] - headings[:-1]
                # Wrap angle differences to [-pi, pi]
                diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
                heading_variations = np.abs(diffs)
                mean_heading_var = float(np.mean(heading_variations))
                self.all_vehicle_mean_heading_variations.append(mean_heading_var)

    def finalize(self) -> Dict[str, float]:
        arr_speed = np.array(self.all_vehicle_mean_variations)
        arr_heading = np.array(self.all_vehicle_mean_heading_variations)
        if arr_speed.size == 0:
            return {
                "other_veh_speed_var_mean": 0.0,
                "other_veh_speed_var_var": 0.0,
                "other_veh_speed_var_max": 0.0,
                "other_veh_heading_var_mean": 0.0,
                "other_veh_heading_var_var": 0.0,
                "other_veh_heading_var_max": 0.0,
            }

        return {
            "other_veh_speed_var_mean": float(np.mean(arr_speed)),
            "other_veh_speed_var_var": float(np.var(arr_speed)),
            "other_veh_speed_var_max": float(np.max(arr_speed)),
            "other_veh_heading_var_mean": float(np.mean(arr_heading)),
            "other_veh_heading_var_var": float(np.var(arr_heading)),
            "other_veh_heading_var_max": float(np.max(arr_heading)),
        }
