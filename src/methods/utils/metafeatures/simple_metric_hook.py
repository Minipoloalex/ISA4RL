import numpy as np
from typing import Dict, List, Any
from .base_metric_hook import BaseMetricHook
from .step_info import StepInfo
from scipy.stats import skew, kurtosis
class SimpleEgoMetricsHook(BaseMetricHook):
    """
    A unified metric hook to track rewards, episode lengths, collisions, 
    timeouts, and speeds for a highway-env evaluation probe.
    Now includes higher-order statistics like SNR, Skewness, and Kurtosis.
    """

    def __init__(self):
        self._reset_stats()

    def _reset_stats(self) -> None:
        # Cross-episode aggregate records
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_crashes: List[int] = []
        self.episode_timeouts: List[int] = []
        self.all_speeds: List[float] = []

        # Current episode state
        self.current_reward = 0.0
        self.current_length = 0
        self.current_crashed = False
        self.current_timeout = False

    def on_probe_start(self) -> None:
        self._reset_stats()

    def on_episode_start(self) -> None:
        self.current_reward = 0.0
        self.current_length = 0
        self.current_crashed = False
        self.current_timeout = False

    def on_step(self, context: StepInfo) -> None:
        self.current_reward += float(context.reward)
        self.current_length += 1

        if context.info.get("crashed", False):
            self.current_crashed = True

        if context.truncated:
            self.current_timeout = True

        speed = context.info["speed"]
        self.all_speeds.append(float(speed))

    def on_episode_end(self) -> None:
        self.episode_rewards.append(self.current_reward)
        self.episode_lengths.append(self.current_length)
        self.episode_crashes.append(1 if self.current_crashed else 0)
        self.episode_timeouts.append(1 if self.current_timeout else 0)

    def finalize(self) -> Dict[str, float]:
        """Aggregate all collected data into summary statistics."""
        
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        speeds = np.array(self.all_speeds)

        def calc_snr(mean_val: float, std_val: float) -> float:
            return float(mean_val / (std_val + 1e-8))

        def safe_stat(func, data: np.ndarray) -> float:
            if len(data) < 2:
                return 0.0
            # nan_policy='omit' ensures one bad step doesn't poison the whole metric
            return float(func(data, nan_policy='omit'))

        r_mean = float(np.mean(rewards))
        r_std = float(np.std(rewards))
        
        s_mean = float(np.mean(speeds))
        s_std = float(np.std(speeds))

        metrics = {
            "reward_mean": r_mean,
            "reward_std": r_std,
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "reward_median": float(np.median(rewards)),
            "reward_snr": calc_snr(r_mean, r_std),
            "reward_skew": safe_stat(skew, rewards),
            "reward_kurtosis": safe_stat(kurtosis, rewards),
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)),
            "collision_rate": float(np.mean(self.episode_crashes)),
            "timeout_rate": float(np.mean(self.episode_timeouts)),
            "speed_mean": s_mean,
            "speed_std": s_std,
            "speed_max": float(np.max(speeds)),
            "speed_median": float(np.median(speeds)),
            "speed_snr": calc_snr(s_mean, s_std),
            "speed_skew": safe_stat(skew, speeds),
            "speed_kurtosis": safe_stat(kurtosis, speeds),
        }

        return metrics
