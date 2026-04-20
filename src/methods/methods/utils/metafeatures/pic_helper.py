import numpy as np
import optuna

def extract_pic_poic(all_scores_per_param, n_trials=200, n_bins=100000, clip_percent=0.0):
    """
    Extracts the Policy Information Capacity (PIC) and Policy Optimal Information Capacity (POIC)
    based on the original paper's implementation.

    Parameters:
    - all_scores_per_param: np.ndarray of shape (N, E), where N is the number of sampled policies,
      and E is the number of episodes evaluated per policy.
    - n_trials: Number of trials for optuna optimization (default: 200).
    - n_bins: Number of bins for histogram discretization used in PIC (default: 100000).
    - clip_percent: Percentile clipping for extreme scores to reduce variance (default: 0.0).

    Returns:
    - pic (float): Policy Information Capacity (mi_r in the original code).
    - poic (float): Policy Optimal Information Capacity (mi_o in the original code).
    - metrics (dict): A dictionary containing additional metrics.
    """
    # Disable optuna logging to keep console clean unless debugging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_scores_per_param = np.array(all_scores_per_param)
    all_mean_scores = all_scores_per_param.mean(axis=1)

    # Optional clipping to remove extreme outliers
    if clip_percent > 0:
        upper = np.percentile(all_mean_scores, 100 - clip_percent)
        lower = np.percentile(all_mean_scores, clip_percent)
        all_scores_per_param = np.clip(all_scores_per_param, lower, upper)

    all_scores = all_scores_per_param.flatten()
    r_max = all_scores.max()

    # --- POIC (Policy Optimal Information Capacity) Calculation ---
    def objective(trial):
        temperature = trial.suggest_float('temperature', 1e-4, 2e4, log=True)
        # Compute P(O=1 | r) and expectations
        p_o1 = np.exp((all_scores - r_max) / temperature).mean()
        p_o1_ts = np.exp((all_scores_per_param - r_max) / temperature).mean(axis=1)
        
        # Calculate marginal and conditional entropy for the optimality variable
        marginal = -p_o1 * np.log(p_o1 + 1e-12) - (1 - p_o1) * np.log(1 - p_o1 + 1e-12)
        conditional = np.mean(-p_o1_ts * np.log(p_o1_ts + 1e-12) - (1 - p_o1_ts) * np.log(1 - p_o1_ts + 1e-12))
        mutual_information = marginal - conditional
        return mutual_information

    # Maximize the mutual information over the temperature hyperparameter
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    trial = study.best_trial
    poic = trial.value
    temperature = trial.params['temperature']
    
    # Calculate POIC entropies for metrics dictionary
    p_o1 = np.exp((all_scores - r_max) / temperature).mean()
    p_o1_ts = np.exp((all_scores_per_param - r_max) / temperature).mean(axis=1)
    poic_marginal_entropy = -p_o1 * np.log(p_o1 + 1e-12) - (1 - p_o1) * np.log(1 - p_o1 + 1e-12)
    poic_conditional_entropy = np.mean(-p_o1_ts * np.log(p_o1_ts + 1e-12) - (1 - p_o1_ts) * np.log(1 - p_o1_ts + 1e-12))

    # --- PIC (Policy Information Capacity) Calculation ---
    hist = np.histogram(all_scores, bins=n_bins)
    discretization_all = hist[0] / len(all_scores)
    pic_marginal_entropy = -np.sum(discretization_all * np.log(discretization_all + 1e-12))
    
    discretization_r_theta = [np.histogram(x, bins=hist[1])[0] / len(x) for x in all_scores_per_param]
    pic_conditional_entropy = -np.mean([np.sum(p_r_theta * np.log(p_r_theta + 1e-12)) for p_r_theta in discretization_r_theta])
    
    pic = pic_marginal_entropy - pic_conditional_entropy

    metrics = {
        'poic_marginal_entropy': poic_marginal_entropy,
        'poic_conditional_entropy': poic_conditional_entropy,
        'poic_temperature': temperature,
        'pic_marginal_entropy': pic_marginal_entropy,
        'pic_conditional_entropy': pic_conditional_entropy,
    }

    return pic, poic, metrics

if __name__ == "__main__":
    # Example Usage:
    # 10 policies, 5 episodes evaluated per policy
    dummy_scores = np.random.randn(10, 5) * 10 + 50
    pic, poic, metrics = extract_pic_poic(dummy_scores, n_trials=50, n_bins=100)
    print(f"Calculated PIC: {pic:.4f}")
    print(f"Calculated POIC: {poic:.4f}")
