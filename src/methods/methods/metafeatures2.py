import hashlib
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, SupportsFloat
import gymnasium as gym
import numpy as np
from methods.utils.metafeatures.metrics import SimpleEgoMetricsHook
from methods.utils.general_utils import as_float

from methods.configs import InstanceConfig
from methods.utils.general_utils import _flatten_obs
from methods.utils.metafeature_utils import (
    Trajectory,
    StepInfo,
    make_random_policy,
    constant_policy,
    default_idle_action,
    ensure_idm_vehicle,
    PolicyFn,
)

INITIAL_GEOMETRY_SAMPLES = 3
MAX_FREE_SPACE_METERS = 200.0
MAX_TTC_SECONDS = 30.0


def extract_metafeatures(config: InstanceConfig):
    before = time.perf_counter()
    env = config.ensure_test_env()
    env_name = env.spec.id
    env_features = _collect_env_features(env_name, env)

    trajectories_random: List[Trajectory] = _run_probe(
        env=env,
        policy=make_random_policy(env),
        episodes=config.n_test_episodes,
    )
    trajectories_idm = _run_idm_probe(
        env=env,
        episodes=config.n_test_episodes,
    )

    # save_trajectories("random", trajectories_random, config)
    # save_trajectories("idm", trajectories_idm, config)
    config.close()
    metafeatures_random = traj_metafeatures("random", trajectories_random)
    metafeatures_idm = traj_metafeatures("idm", trajectories_idm)
    elapsed = time.perf_counter() - before

    idm_advantage = (
        idm_probe["mean_episode_return"] - random_probe["mean_episode_return"]
    )
    safety_delta = random_probe["collision_rate"] - idm_probe["collision_rate"]
    features: Dict[str, float] = {}
    features.update(env_features)
    features.update(_random_probe_features(random_probe))
    features.update(_baseline_probe_features(idm_probe, "idm"))
    features.update(
        _derived_features(
            random_probe=random_probe,
            idm_probe=idm_probe,
        )
    )
    features.update(_action_diversity_features(trajectories_random, trajectories_idm))
    features.update(_state_exploration_features(trajectories_random, trajectories_idm))
    features.update(_temporal_dynamics_features(trajectories_random, trajectories_idm))
    features.update(_spatial_coverage_features(trajectories_random, trajectories_idm))
    features.update(_interaction_metrics(trajectories_random, trajectories_idm))
    return {
        "instance_id": config.id,
        "env_config_id": config.id_env_config,
        "obs_config_id": config.id_obs_config,
        "env_name": env_name,
        "generated_at": time.time_ns(),
        "elapsed_seconds": elapsed,
        "probes": {
            "random": random_probe,
            "idm_like": idm_probe,
        },
        "diagnostics": {
            "idm_advantage": idm_advantage,
            "idm_safety_gain": safety_delta,
        },
        "features": features,
    }

...

def _action_diversity_features(
    random_trajectories: List[Trajectory],
    idm_trajectories: List[Trajectory],
) -> Dict[str, float]:
    """Calculate action diversity metrics for different policies."""
    features: Dict[str, float] = {}
    
    def calc_action_stats(trajectories: List[Trajectory]) -> Dict[str, float]:
        action_counts: Counter = Counter()
        action_entropy = 0.0
        total_actions = 0
        
        for traj in trajectories:
            for step in traj:
                action = step.action
                action_counts[action] += 1
                total_actions += 1
        
        if total_actions == 0:
            return {}
        
        # Calculate entropy
        probs = np.array([action_counts[a] / total_actions for a in action_counts])
        probs = probs[probs > 0]
        action_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Calculate action variance (how spread out actions are)
        mean_action = np.mean(list(action_counts.keys()))
        action_variance = np.mean([(a - mean_action) ** 2 for a in action_counts.keys()])
        
        # Calculate action range
        action_range = max(action_counts.keys()) - min(action_counts.keys()) if action_counts else 0
        
        return {
            "action_entropy": float(action_entropy),
            "action_variance": float(action_variance),
            "action_range": float(action_range),
            "unique_actions": float(len(action_counts)),
            "most_common_action": float(action_counts.most_common(1)[0][0]) if action_counts else 0,
        }
    
    random_action_stats = calc_action_stats(random_trajectories)
    idm_action_stats = calc_action_stats(idm_trajectories)
    
    features.update(random_action_stats)
    features.update(idm_action_stats)
    
    # Compare action diversity between policies
    features["action_diversity_gap"] = random_action_stats.get("action_entropy", 0) - idm_action_stats.get("action_entropy", 0)
    features["action_entropy_ratio"] = random_action_stats.get("action_entropy", 0) / (idm_action_stats.get("action_entropy", 1e-10) + 1e-10)
    
    return features


def _state_exploration_features(
    random_trajectories: List[Trajectory],
    idm_trajectories: List[Trajectory],
) -> Dict[str, float]:
    """Calculate state exploration metrics."""
    features: Dict[str, float] = {}
    
    def calc_exploration_stats(trajectories: List[Trajectory]) -> Dict[str, float]:
        visited_states: set = set()
        state_transitions: Counter = Counter()
        total_steps = 0
        
        for traj in trajectories:
            prev_state = None
            for step in traj:
                # Create a simplified state representation
                obs = step.obs
                flat_obs = _flatten_obs(obs)
                state_key = tuple(np.round(flat_obs, 6).astype(int).tolist())
                
                if prev_state is not None:
                    state_transitions[(prev_state, state_key)] += 1
                
                visited_states.add(state_key)
                prev_state = state_key
                total_steps += 1
        
        if total_steps == 0:
            return {}
        
        # Calculate exploration metrics
        unique_states = len(visited_states)
        state_coverage = unique_states / max(total_steps, 1)
        
        # Calculate transition entropy (how diverse the transitions are)
        transition_probs = np.array([count / total_steps for count in state_transitions.values()])
        transition_entropy = -np.sum(transition_probs * np.log2(transition_probs + 1e-10))
        
        # Calculate average transition frequency
        avg_transition_freq = np.mean(list(state_transitions.values()))
        
        return {
            "unique_states_visited": float(unique_states),
            "state_coverage": float(state_coverage),
            "transition_entropy": float(transition_entropy),
            "avg_transition_frequency": float(avg_transition_freq),
        }
    
    random_exploration = calc_exploration_stats(random_trajectories)
    idm_exploration = calc_exploration_stats(idm_trajectories)
    
    features.update(random_exploration)
    features.update(idm_exploration)
    
    # Compare exploration between policies
    features["exploration_gap"] = random_exploration.get("unique_states_visited", 0) - idm_exploration.get("unique_states_visited", 0)
    features["exploration_efficiency_ratio"] = random_exploration.get("state_coverage", 0) / (idm_exploration.get("state_coverage", 1e-10) + 1e-10)
    
    return features


def _temporal_dynamics_features(
    random_trajectories: List[Trajectory],
    idm_trajectories: List[Trajectory],
) -> Dict[str, float]:
    """Calculate temporal dynamics metrics."""
    features: Dict[str, float] = {}
    
    def calc_temporal_stats(trajectories: List[Trajectory]) -> Dict[str, float]:
        state_changes = 0
        total_steps = 0
        speed_changes = 0
        reward_changes = 0
        
        for traj in trajectories:
            prev_state = None
            prev_speed = None
            prev_reward = None
            
            for step in traj:
                obs = step.obs
                flat_obs = _flatten_obs(obs)
                current_state = tuple(np.round(flat_obs, 6).astype(int).tolist())
                
                if prev_state is not None and current_state != prev_state:
                    state_changes += 1
                
                if prev_speed is not None and step.info.get("speed") != prev_speed:
                    speed_changes += 1
                
                if prev_reward is not None and step.reward != prev_reward:
                    reward_changes += 1
                
                prev_state = current_state
                prev_speed = step.info.get("speed")
                prev_reward = step.reward
                total_steps += 1
        
        if total_steps == 0:
            return {}
        
        return {
            "state_change_rate": float(state_changes / total_steps),
            "speed_change_rate": float(speed_changes / total_steps),
            "reward_change_rate": float(reward_changes / total_steps),
            "avg_steps_per_episode": float(total_steps / len(trajectories)) if trajectories else 0,
        }
    
    random_temporal = calc_temporal_stats(random_trajectories)
    idm_temporal = calc_temporal_stats(idm_trajectories)
    
    features.update(random_temporal)
    features.update(idm_temporal)
    
    # Compare temporal dynamics
    features["temporal_stability_gap"] = random_temporal.get("state_change_rate", 0) - idm_temporal.get("state_change_rate", 0)
    features["dynamics_consistency_ratio"] = random_temporal.get("state_change_rate", 0) / (idm_temporal.get("state_change_rate", 1e-10) + 1e-10)
    
    return features


def _spatial_coverage_features(
    random_trajectories: List[Trajectory],
    idm_trajectories: List[Trajectory],
) -> Dict[str, float]:
    """Calculate spatial coverage metrics."""
    features: Dict[str, float] = {}
    
    def calc_spatial_stats(trajectories: List[Trajectory]) -> Dict[str, float]:
        positions: List[float] = []
        lanes_visited: set = set()
        total_distance = 0.0
        
        for traj in trajectories:
            prev_pos = None
            for step in traj:
                pos = step.info.get("position")
                if pos is not None:
                    positions.append(float(pos[0]))
                    if prev_pos is not None:
                        total_distance += abs(float(pos[0]) - prev_pos[0])
                    prev_pos = pos
                
                lane = step.info.get("lane")
                if lane is not None:
                    lanes_visited.add(lane)
        
        if not positions:
            return {}
        
        # Calculate spatial metrics
        min_pos = min(positions)
        max_pos = max(positions)
        range_covered = max_pos - min_pos
        
        # Calculate velocity statistics
        velocities = []
        for i in range(1, len(positions)):
            dt = 1.0  # Assuming 1 second per step
            vel = (positions[i] - positions[i-1]) / dt
            velocities.append(vel)
        
        avg_velocity = np.mean(velocities) if velocities else 0
        velocity_std = np.std(velocities) if velocities else 0
        
        return {
            "range_covered_m": float(range_covered),
            "lanes_visited": float(len(lanes_visited)),
            "avg_velocity_mps": float(avg_velocity),
            "velocity_std_mps": float(velocity_std),
            "total_distance_m": float(total_distance),
        }
    
    random_spatial = calc_spatial_stats(random_trajectories)
    idm_spatial = calc_spatial_stats(idm_trajectories)
    
    features.update(random_spatial)
    features.update(idm_spatial)
    
    # Compare spatial coverage
    features["spatial_coverage_gap"] = random_spatial.get("range_covered_m", 0) - idm_spatial.get("range_covered_m", 0)
    features["lane_diversity_ratio"] = random_spatial.get("lanes_visited", 0) / (idm_spatial.get("lanes_visited", 1e-10) + 1e-10)
    
    return features


def _interaction_metrics(
    random_trajectories: List[Trajectory],
    idm_trajectories: List[Trajectory],
) -> Dict[str, float]:
    """Calculate vehicle interaction metrics."""
    features: Dict[str, float] = {}
    
    def calc_interaction_stats(trajectories: List[Trajectory]) -> Dict[str, float]:
        interactions = 0
        close_encounters = 0
        lane_changes = 0
        total_steps = 0
        
        for traj in trajectories:
            prev_lane = None
            for step in traj:
                lane = step.info.get("lane")
                if lane is not None:
                    if prev_lane is not None and lane != prev_lane:
                        lane_changes += 1
                    
                    # Count interactions (other vehicles nearby)
                    if step.info.get("nearby_vehicles", 0) > 0:
                        interactions += step.info.get("nearby_vehicles", 0)
                    
                    # Count close encounters (TTC < 2 seconds)
                    if step.info.get("min_ttc", float('inf')) < 2.0:
                        close_encounters += 1
                
                prev_lane = lane
                total_steps += 1
        
        if total_steps == 0:
            return {}
        
        return {
            "interactions_per_step": float(interactions / total_steps),
            "close_encounters_per_step": float(close_encounters / total_steps),
            "lane_changes_per_step": float(lane_changes / total_steps),
            "interaction_density": float(interactions / len(trajectories)) if trajectories else 0,
        }
    
    random_interaction = calc_interaction_stats(random_trajectories)
    idm_interaction = calc_interaction_stats(idm_trajectories)
    
    features.update(random_interaction)
    features.update(idm_interaction)
    
    # Compare interaction metrics
    features["interaction_reduction"] = random_interaction.get("interactions_per_step", 0) - idm_interaction.get("interactions_per_step", 0)
    features["safety_interaction_ratio"] = random_interaction.get("close_encounters_per_step", 0) / (idm_interaction.get("close_encounters_per_step", 1e-10) + 1e-10)
    
    return features

...
