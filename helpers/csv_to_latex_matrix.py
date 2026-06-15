import csv
import io
import argparse


def latex_matrix(data: str, label: str):
    """
    label is used for the matrix equation. Later adds "eq:" as a prefix
    """
    
    reader = csv.reader(io.StringIO(data.strip()))
    header = next(reader)
    # Remove 'feature_' prefix and replace underscores with hyphens
    features = [f.replace('feature_', '').replace('_', '-') for f in header[1:]]

    z1 = next(reader)[1:]
    z2 = next(reader)[1:]

    matrix_rows = [f"{float(v1):.2f} & {float(v2):.2f} \\\\" for v1, v2 in zip(z1, z2)]
    matrix_str = '\n'.join(matrix_rows)

    features_tex = [f"\\mf{{{f}}} \\\\" for f in features]
    # Remove trailing slashes for the last row
    features_tex[-1] = features_tex[-1].replace(' \\\\', '') 
    matrix_str = matrix_str.replace(' \\\\', ' \\\\', len(matrix_rows)-1) 
    features_str = '\n'.join(features_tex)

    latex = f"""
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
    \\label{{eq:{label}}}
    \\end{{equation}}
    """
    return latex

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label used for the matrix equation. Later adds \"eq:\" as a prefix)",
    )
    args = parser.parse_args()
    EQ_LABEL = args.label

    DATA = """
    Row,feature_lanes_count,feature_random_collision_rate,feature_baseline_obs_volatility_mean,feature_baseline_other_veh_speed_delta_mean,feature_baseline_other_veh_heading_delta_mean,feature_action_state_discontinuity_mean
    Z_{1},-0.012314,-0.705,-0.03415,0.2358,0.3457,-0.0366
    Z_{2},-0.03055,0.1053,0.651,0.02417,-0.01265,-0.403
    """
    ans = latex_matrix(DATA, EQ_LABEL)
    print(ans)
