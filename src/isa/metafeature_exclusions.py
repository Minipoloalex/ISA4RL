"""Metafeature columns intentionally kept out of the ISA metadata dataset."""

from typing import Iterable

# These names are the final CSV column names produced by src/isa/main.py.
# The raw extracted metafeatures remain available in metafeatures.json.
EXCLUDED_METAFEATURE_COLUMNS = {
    "action_space_size",
    "max_steps",
    "obs_categorical_ratio",
    "obs_numerical_ratio",
    "obs_binary_ratio",
    "obs_space_dim_log",
}
EXCLUDE_METAFEATURES_BOTH_PROBES = {
    "mean_reward",
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
for f in EXCLUDE_METAFEATURES_BOTH_PROBES:
    EXCLUDED_METAFEATURE_COLUMNS.add(f"random_{f}")
    EXCLUDED_METAFEATURE_COLUMNS.add(f"baseline_{f}")

def features(feats: Iterable[str]):
    return {f"feature_{f}" for f in feats}

EXCLUDED_METAFEATURE_COLUMNS = features(EXCLUDED_METAFEATURE_COLUMNS)

DOMAIN_SPECIFIC_METAFEATURES = features({
    "lanes_count",
    "traffic_density",
    "random_collision_rate",
    "random_close_collision_rate",
    "random_timeout_rate",
    "random_speed_snr",
    "random_speed_skew",
    "random_speed_kurtosis",
    "random_other_veh_speed_var_mean",
    "random_other_veh_speed_var_var",
    "random_other_veh_speed_var_max",
    "random_other_veh_heading_var_mean",
    "random_other_veh_heading_var_var",
    "random_other_veh_heading_var_max",
    "baseline_collision_rate",
    "baseline_close_collision_rate",
    "baseline_timeout_rate",
    "baseline_speed_snr",
    "baseline_speed_skew",
    "baseline_speed_kurtosis",
    "baseline_other_veh_speed_var_mean",
    "baseline_other_veh_speed_var_var",
    "baseline_other_veh_speed_var_max",
    "baseline_other_veh_heading_var_mean",
    "baseline_other_veh_heading_var_var",
    "baseline_other_veh_heading_var_max",
})
