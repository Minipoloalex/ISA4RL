"""Metafeature columns intentionally kept out of the ISA metadata dataset."""

# These names are the final CSV column names produced by src/isa/main.py.
# The raw extracted metafeatures remain available in metafeatures.json.
EXCLUDED_METAFEATURE_COLUMNS = {
    "feature_random_mean_reward",
    "feature_baseline_mean_reward",
}
