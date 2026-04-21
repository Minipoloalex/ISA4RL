import numpy as np
import gymnasium as gym

# Import the agent and sampler logic directly from the original paper's codebase
from pic.algos import NumpyAgent
from pic.sampler.sampler import run_episode

# Import the metafeature extractor we just created
from methods.utils.metafeatures.pic_helper import extract_pic_poic

def compute_pic_end_to_end(
    env: gym.Env,
    n_samples=1000,
    n_episodes=10,
    n_hidden_layers=2,
    n_hidden_units=64,
    random_dist="normal",
    env_name="",
):
    """
    Computes PIC and POIC end-to-end for an already instantiated gym environment.

    This maps directly to the paper's process:
    1. It initializes the NumpyAgent (MLP) from the original repo.
    2. For `n_samples`, it samples a new set of weights.
    3. For each set of weights, it runs `n_episodes` episodes on the environment.
    4. It passes the resulting (N, E) score matrix to `extract_pic_poic`.

    Parameters:
    - env: Instantiated Gym environment.
    - n_samples (int): Number of random policies to sample (default: 1000).
    - n_episodes (int): Number of episodes to evaluate per policy (default: 10).
    - n_hidden_layers (int): Depth of the random MLP (default: 2).
    - n_hidden_units (int): Width of the random MLP (default: 64).
    - random_dist (str): Distribution for weight sampling ('normal', 'uniform', etc).

    Returns:
    - pic (float): Policy Information Capacity.
    - poic (float): Policy Optimal Information Capacity.
    - metrics (dict): Entropy metrics and best optuna temperature.
    - all_scores (np.ndarray): The raw (N, E) score matrix collected.
    """

    # Instantiate the agent using the original implementation
    agent = NumpyAgent(
        env=env,
        n_hidden_layers=n_hidden_layers,
        n_hidden_units=n_hidden_units,
        random_dist=random_dist,
        env_name=env_name,
    )

    all_scores_per_param = []

    # 3. Sampling loop: Evaluate n_samples policies, n_episodes times each
    for samp_num in range(n_samples):
        # Optional: Print progress every 10%
        if n_samples >= 10 and samp_num % (n_samples // 10) == 0:
            print(f"Sampling policy {samp_num}/{n_samples}...")

        score_episodes = []
        for _ in range(n_episodes):
            score = run_episode(env, agent)
            score_episodes.append(score)

        all_scores_per_param.append(score_episodes)

        # Re-initialize weights for the next random policy
        agent.init_weights()

    all_scores_per_param = np.array(all_scores_per_param)

    # 4. Extract PIC and POIC using our metafeature wrapper
    print("Computing metrics via Optuna optimization...")
    pic, poic, metrics = extract_pic_poic(all_scores_per_param)

    return pic, poic, metrics, all_scores_per_param

