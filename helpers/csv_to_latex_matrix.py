import csv
import io

DATA = """
Row,feature_traffic_density,feature_random_reward_kurtosis,feature_random_obs_volatility_max,feature_random_pca_components_95_var,feature_random_obs_entropy_mean,feature_baseline_reward_skew,feature_baseline_reward_kurtosis,feature_baseline_speed_kurtosis,feature_baseline_other_veh_speed_var_var,feature_action_state_discontinuity_p95
Z_{1},-0.2925,-0.0463,0.378,0.165,0.1053,-0.1051,0.04352,0.06256,-0.01768,0.3733
Z_{2},0.06506,0.0794,-0.2345,0.2588,-0.3428,-0.1736,0.12177,0.1697,0.2932,0.03253
"""
EQ_LABEL = "cont_matrix"

reader = csv.reader(io.StringIO(DATA.strip()))
header = next(reader)
# Remove 'feature_' prefix and replace underscores with hyphens
features = [f.replace('feature_', '').replace('_', '-') for f in header[1:]]

z1 = next(reader)[1:]
z2 = next(reader)[1:]

matrix_rows = [f"{float(v1):.2f} & {float(v2):.2f} \\\\" for v1, v2 in zip(z1, z2)]
matrix_str = '\n'.join(matrix_rows)

features_tex = [f"\\texttt{{{f}}} \\\\" for f in features]
# Remove trailing slashes for the last row
features_tex[-1] = features_tex[-1].replace(' \\\\', '') 
matrix_str = matrix_str.replace(' \\\\', ' \\\\', len(matrix_rows)-1) 
features_str = '\n'.join(features_tex)

latex = f"""% Projection matrix
\\begin{{equation}}
\\begin{{bmatrix}}
Z_1\\\\
Z_2
\\end{{bmatrix}}
=
\\begin{{bmatrix}}
{matrix_str}
\\end{{bmatrix}}^{{T}}
\\begin{{bmatrix}}
{features_str}
\\end{{bmatrix}}
\\label{{eq:{EQ_LABEL}}}
\\end{{equation}}"""

print(latex)
