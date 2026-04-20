import numpy as np

# Import the agent and sampler logic directly from the original paper's codebase
from pic.algos import NumpyAgent
from pic.sampler.sampler import run_episode

# Import the metafeature extractor we just created
from src.methods.methods.utils.metafeatures.pic_metafeature import extract_pic_poic


def compute_pic_end_to_end(
    env,
    n_samples=1000,
    n_episodes=10,
    max_episode_steps=None,
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
    - max_episode_steps (int): Max steps per episode. If None, infers from env.spec.
    - n_hidden_layers (int): Depth of the random MLP (default: 2).
    - n_hidden_units (int): Width of the random MLP (default: 64).
    - random_dist (str): Distribution for weight sampling ('normal', 'uniform', etc).

    Returns:
    - pic (float): Policy Information Capacity.
    - poic (float): Policy Optimal Information Capacity.
    - metrics (dict): Entropy metrics and best optuna temperature.
    - all_scores (np.ndarray): The raw (N, E) score matrix collected.
    """
    # 1. Infer max episode steps if not provided
    if max_episode_steps is None:
        if hasattr(env, "spec") and env.spec is not None and hasattr(env.spec, "max_episode_steps"):
            max_episode_steps = env.spec.max_episode_steps
        else:
            max_episode_steps = 1000  # fallback limit

    # 2. Instantiate the agent using the original implementation
    # Note: NumpyAgent natively calls env.reset() in its __init__ to figure out input size.
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
            score = run_episode(env, agent, max_episode_steps)
            score_episodes.append(score)

        all_scores_per_param.append(score_episodes)

        # Re-initialize weights for the next random policy
        agent.init_weights()

    all_scores_per_param = np.array(all_scores_per_param)

    # 4. Extract PIC and POIC using our metafeature wrapper
    print("Computing metrics via Optuna optimization...")
    pic, poic, metrics = extract_pic_poic(all_scores_per_param)

    return pic, poic, metrics, all_scores_per_param


if __name__ == "__main__":
    # Example usage script
    import gym
    import warnings

    warnings.filterwarnings("ignore")

    try:
        # Create a simple test environment
        test_env_name = "CartPole-v0"
        env = gym.make(test_env_name)

        print(f"--- Running End-to-End Test on {test_env_name} ---")
        # We use small sample numbers for a quick test;
        # For actual usage, use n_samples=10000, n_episodes=10 as in the paper.
        test_samples = 50
        test_episodes = 5

        pic_val, poic_val, metrics_dict, scores = compute_pic_end_to_end(
            env=env,
            n_samples=test_samples,
            n_episodes=test_episodes,
            env_name=test_env_name,
        )

        print("\n--- Results ---")
        print(f"Collected scores shape: {scores.shape} (Policies x Episodes)")
        print(f"PIC: {pic_val:.4f}")
        print(f"POIC: {poic_val:.4f}")

    except Exception as e:
        print(f"Test failed. Ensure Gym is installed correctly. Error: {e}")
