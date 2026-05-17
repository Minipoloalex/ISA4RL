"""Metafeature columns intentionally kept out of the ISA metadata dataset."""

# These names are the final CSV column names produced by src/isa/main.py.
# The raw extracted metafeatures remain available in metafeatures.json.
EXCLUDED_METAFEATURE_COLUMNS = {
    "random_mean_reward",
    "baseline_mean_reward",
    "action_space_size",
    "max_steps",
    "obs_categorical_ratio",
    "obs_numerical_ratio",
    "obs_binary_ratio",
    "speed_mean",
    "speed_std",
    "speed_max",
    "speed_median",
    "positive_reward_rate",
    "length_mean",
    "length_std",
    "reward_std",
    "reward_min",
    "reward_max",
    "reward_median",
    "obs_mean",
    "obs_std",
    "obs_sparsity",
}

EXCLUDED_METAFEATURE_COLUMNS = {
    f"feature_{f}" for f in EXCLUDED_METAFEATURE_COLUMNS
}
