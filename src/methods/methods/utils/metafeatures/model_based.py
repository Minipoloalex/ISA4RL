import numpy as np
import gymnasium as gym
from sklearn.linear_model import LinearRegression
from collections import Counter
import logging
from methods.utils.metafeature_utils import safe_copy_env
from methods.utils.general_utils import _flatten_obs

logger = logging.getLogger(__name__)


def _action_distance(action_space: gym.Space, action_1, action_2) -> float:
    if isinstance(action_space, gym.spaces.Discrete):
        return 0.0 if int(action_1) == int(action_2) else 1.0

    if isinstance(action_space, gym.spaces.MultiDiscrete):
        action_1_arr = np.asarray(action_1, dtype=int).ravel()
        action_2_arr = np.asarray(action_2, dtype=int).ravel()
        return float(np.linalg.norm(action_1_arr != action_2_arr))

    if isinstance(action_space, gym.spaces.MultiBinary):
        action_1_arr = np.asarray(action_1, dtype=int).ravel()
        action_2_arr = np.asarray(action_2, dtype=int).ravel()
        return float(np.linalg.norm(action_1_arr != action_2_arr))

    if isinstance(action_space, gym.spaces.Box):
        action_1_arr = np.asarray(action_1, dtype=np.float32).ravel()
        action_2_arr = np.asarray(action_2, dtype=np.float32).ravel()
        action_delta = action_1_arr - action_2_arr

        low = np.asarray(action_space.low, dtype=np.float32).ravel()
        high = np.asarray(action_space.high, dtype=np.float32).ravel()
        width = high - low
        finite_width = np.isfinite(width) & (np.abs(width) > 1e-8)
        if np.all(finite_width):
            action_delta = action_delta / width

        return float(np.linalg.norm(action_delta))

    raise ValueError(f"Unsupported action space type: {type(action_space)}")


def _action_features(action_space: gym.Space, action) -> np.ndarray:
    if isinstance(action_space, gym.spaces.Discrete):
        encoded_action = np.zeros(action_space.n, dtype=np.float32)
        encoded_action[int(action)] = 1.0
        return encoded_action

    if isinstance(action_space, gym.spaces.MultiDiscrete):
        action_arr = np.asarray(action, dtype=int).ravel()
        encoded_parts = []
        for value, n_values in zip(action_arr, action_space.nvec.ravel()):
            encoded = np.zeros(int(n_values), dtype=np.float32)
            encoded[int(value)] = 1.0
            encoded_parts.append(encoded)
        return np.concatenate(encoded_parts)

    if isinstance(action_space, gym.spaces.MultiBinary):
        return np.asarray(action, dtype=np.float32).ravel()

    if isinstance(action_space, gym.spaces.Box):
        action_arr = np.asarray(action, dtype=np.float32).ravel()
        low = np.asarray(action_space.low, dtype=np.float32).ravel()
        high = np.asarray(action_space.high, dtype=np.float32).ravel()
        width = high - low
        finite_width = np.isfinite(width) & (np.abs(width) > 1e-8)
        if np.all(finite_width):
            return (action_arr - low) / width
        return action_arr

    raise ValueError(f"Unsupported action space type: {type(action_space)}")


def _sample_action_pairs(action_space: gym.Space, num_pairs: int):
    if isinstance(action_space, gym.spaces.Discrete):
        action_pairs = [
            (action_1, action_2)
            for action_1 in range(action_space.n)
            for action_2 in range(action_1 + 1, action_space.n)
        ]
        if len(action_pairs) <= num_pairs:
            return action_pairs

        selected_indices = np.random.choice(len(action_pairs), size=num_pairs, replace=False)
        return [action_pairs[int(index)] for index in selected_indices]

    return [
        (action_space.sample(), action_space.sample())
        for _ in range(num_pairs)
    ]


def _next_action(action_space: gym.Space, current_action, step_size: float):
    if isinstance(action_space, gym.spaces.Box):
        noise = np.random.normal(0, step_size, size=current_action.shape)
        return np.clip(
            current_action + noise,
            action_space.low,
            action_space.high,
        )

    return action_space.sample()


def estimate_normalized_lipschitz(
    env: gym.Env,
    num_states=50,
    action_pairs_per_state=20,
    reward_samples_per_state=15,
):
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
        
        base_env = safe_copy_env(env)

        sample_rewards = []
        for _ in range(reward_samples_per_state):
            temp_env = safe_copy_env(base_env)
            _, r, _, _, _ = temp_env.step(env.action_space.sample())
            sample_rewards.append(r)
            temp_env.close()
        
        reward_std = np.std(sample_rewards)
        if reward_std < 1e-6:
            reward_std = 1.0 
        
        action_pairs = _sample_action_pairs(env.action_space, action_pairs_per_state)
        for a1, a2 in action_pairs:
            dist = _action_distance(env.action_space, a1, a2)
            
            if dist < 1e-6:
                continue
                
            env_copy1 = safe_copy_env(base_env)
            _, r1, _, _, _ = env_copy1.step(a1)
            
            env_copy2 = safe_copy_env(base_env)
            _, r2, _, _, _ = env_copy2.step(a2)
            
            ratio = (abs(r1 - r2) / reward_std) / dist
            
            if ratio > max_lipschitz:
                max_lipschitz = ratio
            
            env_copy1.close()
            env_copy2.close()

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
                
        base_env = safe_copy_env(env)
        action = env.action_space.sample()
        
        next_states = []
        for _ in range(trials_per_action):
            temp_env = safe_copy_env(base_env)
            next_obs, _, _, _, _ = temp_env.step(action)
            next_states.append(_flatten_obs(next_obs))
            temp_env.close()
            
        next_states = np.array(next_states)
        state_variance = np.var(next_states, axis=0).mean()
        variances.append(state_variance)
        base_env.close()

    return float(np.mean(variances))


def compute_transition_linearity(env, num_samples=1000):
    O_current = []
    O_next = []
    A = []
    
    obs, _ = env.reset()
    
    for _ in range(num_samples):
        action = env.action_space.sample()
        
        flat_obs = _flatten_obs(obs)
        flat_action = _action_features(env.action_space, action)
        
        next_obs, _, terminated, truncated, _ = env.step(action)
        flat_next_obs = _flatten_obs(next_obs)
        
        O_current.append(flat_obs)
        A.append(flat_action)
        O_next.append(flat_next_obs)
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    O_current = np.array(O_current)
    O_next = np.array(O_next)
    A = np.array(A)
    
    # If the observation space is very high-dimensional (e.g. images),
    # reduce dimensionality using PCA to avoid perfectly overfitting
    # the Linear Regression model (which happens when features > samples).
    if O_current.shape[1] > 100:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(64, num_samples // 2))
        # Fit on both current and next to cover the whole state distribution we've seen
        O_combined = np.vstack([O_current, O_next])
        pca.fit(O_combined)
        O_current = pca.transform(O_current)
        O_next = pca.transform(O_next)
        
    X = np.hstack([O_current, A])
    Y = O_next
    
    model = LinearRegression()
    model.fit(X, Y)
    
    return float(model.score(X, Y))



def compute_action_landscape_ruggedness(env, num_states=20, walk_length=50, max_episode_steps=None, step_size=0.1):
    autocorrelations = []

    for _ in range(num_states):
        env.reset()
        for _ in range(np.random.randint(1, 10)):   # max_episode_steps // 2
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()

        base_env = safe_copy_env(env)
        current_action = env.action_space.sample()
        rewards = []

        for _ in range(walk_length):
            temp_env = safe_copy_env(base_env)
            _, r, _, _, _ = temp_env.step(current_action)
            rewards.append(r)
            temp_env.close()

            current_action = _next_action(env.action_space, current_action, step_size)

        if np.std(rewards) < 1e-6:
            autocorrelations.append(1.0)
            continue

        r_t = rewards[:-1]
        r_t1 = rewards[1:]
        correlation = np.corrcoef(r_t, r_t1)[0, 1]
        
        if np.isnan(correlation):
            correlation = 1.0
            
        autocorrelations.append(correlation)
        base_env.close()

    mean_autocorr = np.mean(autocorrelations)
    
    ruggedness = 1.0 - mean_autocorr
    return ruggedness


def compute_state_entropy(env: gym.Env, num_steps=10000, decimals=1):
    state, _ = env.reset()
    state_counts = Counter()

    for _ in range(num_steps):
        state_rep = tuple(np.round(_flatten_obs(state), decimals))

        state_counts[state_rep] += 1
        
        action = env.action_space.sample()
        state, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            state, _ = env.reset()

    counts = np.array(list(state_counts.values()))
    probabilities = counts / num_steps

    return float(-np.sum(probabilities * np.log2(probabilities)))
