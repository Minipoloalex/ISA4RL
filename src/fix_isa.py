import pandas as pd
import numpy as np

path = "results/isa/old_instancespace_dataset.csv"
new_path = "results/isa/new_instancespace_dataset.csv"

df = pd.read_csv(path)

feature_cols = [c for c in df.columns if c.startswith("feature_")]
algo_cols = [c for c in df.columns if c.startswith("algo_")]

# Useful debugging summary
summary = pd.DataFrame({
    "missing": df[feature_cols].isna().mean() * 100,
    "min": df[feature_cols].min(),
    "max": df[feature_cols].max(),
    "n_unique": df[feature_cols].nunique(),
})

problematic = summary.query(
    "missing > 0 or n_unique <= 1 or min <= 0"
)


problematic_features = list(problematic.index)
potential_useful_features = list(filter(lambda x : x not in problematic_features, feature_cols))
lean_features = list(filter(lambda x : "sticky" not in x, potential_useful_features))

print()
print(len(problematic), "problematic features detected")
print(len(summary), "all features detected")
print(len(summary) - len(problematic), "potential useful features detected")
print(len(lean_features), "potential usable features")
print()
print("Potential Useful Features:")
for feature in lean_features:
    print(f"  {feature}")
print()
# print(lean_features)

selected_features = [
"feature_random_obs_std",
"feature_random_reward_return_var_early10_frac",
"feature_idm_mean_speed",
"feature_idm_return_p90",
"feature_idm_return_per_step",
"feature_obs_noise_return_p90",
]

new_df = df[["instances"] + lean_features + algo_cols] # TODO: filter by lean features
new_df = df[["instances"] + selected_features + algo_cols] # TODO: this allows saving results/isa

new_df.to_csv(new_path, index=False)
