"""Metafeature columns intentionally kept out of the ISA metadata dataset."""

from typing import Iterable

# These names are the final CSV column names produced by src/isa/main.py.
# The raw extracted metafeatures remain available in metafeatures.json.
EXCLUDED_METAFEATURE_COLUMNS = {
    "action_space_size",
    "max_steps",
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
    
})
