import numpy as np
from copy import deepcopy
import gymnasium as gym
from sklearn.linear_model import LinearRegression
from collections import Counter
import logging


logger = logging.getLogger(__name__)


def estimate_normalized_lipschitz(env: gym.Env, num_states=50, action_pairs_per_state=20):
    """
    Estimates the normalized Lipschitz constant of the reward function.
    
    This measures how much the reward can change for a given change in action.
    Higher values suggest the reward landscape is more sensitive/variable.
    """
    max_lipschitz = 0.0

    for _ in range(num_states):
        env.reset()
        
        # Burn-in period to reach a typical state
        for _ in range(np.random.randint(1, 10)):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset()
            
        base_env = deepcopy(env)
        
        sample_rewards = []
        for _ in range(15):
            temp_env = deepcopy(base_env)
            _, r, _, _, _ = temp_env.step(env.action_space.sample())
            sample_rewards.append(r)
        
        reward_std = np.std(sample_rewards)
        if reward_std < 1e-6:
            reward_std = 1.0 
        
        for _ in range(action_pairs_per_state):
            a1 = env.action_space.sample()
            a2 = env.action_space.sample()
            
            dist = np.linalg.norm(a1 - a2)
            
            if dist < 1e-6:
                continue
                
            env_copy1 = deepcopy(base_env)
            _, r1, _, _, _ = env_copy1.step(a1)
            
            env_copy2 = deepcopy(base_env)
            _, r2, _, _, _ = env_copy2.step(a2)
            
            ratio = (abs(r1 - r2) / reward_std) / dist
            
            if ratio > max_lipschitz:
                max_lipschitz = ratio

    return max_lipschitz


def compute_transition_stochasticity(env, num_states=30, trials_per_action=10):
    variances = []
    
    for _ in range(num_states):
        env.reset()
        for _ in range(np.random.randint(1, 10)):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
                
        base_env = deepcopy(env)
        action = env.action_space.sample()
        
        next_states = []
        for _ in range(trials_per_action):
            temp_env = deepcopy(base_env)
            next_obs, _, _, _, _ = temp_env.step(action)
            next_states.append(next_obs)
            
        next_states = np.array(next_states)
        state_variance = np.var(next_states, axis=0).mean()
        variances.append(state_variance)

    return float(np.mean(variances))


def compute_transition_linearity(env, num_samples=1000):
    X = []
    Y = []
    
    obs, _ = env.reset()
    
    for _ in range(num_samples):
        action = env.action_space.sample()
        
        flat_obs = np.atleast_1d(obs).flatten()
        flat_action = np.atleast_1d(action).flatten()
        
        x_vector = np.concatenate([flat_obs, flat_action])
        
        next_obs, _, terminated, truncated, _ = env.step(action)
        
        y_vector = np.atleast_1d(next_obs).flatten()
        
        X.append(x_vector)
        Y.append(y_vector)
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    X = np.array(X)
    Y = np.array(Y)
    
    model = LinearRegression()
    model.fit(X, Y)
    
    return float(model.score(X, Y))


def compute_action_landscape_ruggedness(env, num_states=20, walk_length=50, step_size=0.1):
    autocorrelations = []

    for _ in range(num_states):
        env.reset()
        for _ in range(np.random.randint(1, 10)):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()

        base_env = deepcopy(env)
        current_action = env.action_space.sample()
        rewards = []

        for _ in range(walk_length):
            temp_env = deepcopy(base_env)
            _, r, _, _, _ = temp_env.step(current_action)
            rewards.append(r)

            if hasattr(env.action_space, 'high'):
                noise = np.random.normal(0, step_size, size=current_action.shape)
                current_action = np.clip(
                    current_action + noise, 
                    env.action_space.low, 
                    env.action_space.high
                )
            else:
                current_action = env.action_space.sample()

        if np.std(rewards) < 1e-6:
            autocorrelations.append(1.0)
            continue

        r_t = rewards[:-1]
        r_t1 = rewards[1:]
        correlation = np.corrcoef(r_t, r_t1)[0, 1]
        
        if np.isnan(correlation):
            correlation = 1.0
            
        autocorrelations.append(correlation)

    mean_autocorr = np.mean(autocorrelations)
    
    ruggedness = 1.0 - mean_autocorr
    return ruggedness


def compute_state_entropy(env: gym.Env, num_steps=10000, decimals=1):
    state, _ = env.reset()
    state_counts = Counter()

    for _ in range(num_steps):
        if isinstance(state, np.ndarray):
            state_rep = tuple(np.round(state, decimals).flatten())
        else:
            state_rep = state

        state_counts[state_rep] += 1
        
        action = env.action_space.sample()
        state, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            state, _ = env.reset()

    counts = np.array(list(state_counts.values()))
    probabilities = counts / num_steps

    return float(-np.sum(probabilities * np.log2(probabilities)))
