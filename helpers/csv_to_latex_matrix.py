import csv
import io
import argparse
from pathlib import Path


def latex_matrix(data: str, label: str):
    """
    label is used for the matrix equation. Later adds "eq:" as a prefix
    """
    
    reader = csv.reader(io.StringIO(data.strip()))
    header = next(reader)
    # Remove "feature_" prefix and replace underscores with hyphens
    features = [f.replace("feature_", "").replace("_", "-") for f in header[1:]]

    z1 = next(reader)[1:]
    z2 = next(reader)[1:]

    matrix_rows = [f"{float(v1):.2f} & {float(v2):.2f}" for v1, v2 in zip(z1, z2)]
    matrix_rows = [
        f"{row} \\\\" if row_index < len(matrix_rows) - 1 else row
        for row_index, row in enumerate(matrix_rows)
    ]
    matrix_str = "\n".join(matrix_rows)

    features_tex = [f"\\mf{{{f}}} \\\\" for f in features]
    # Remove trailing slashes for the last row
    features_tex[-1] = features_tex[-1].replace(" \\\\", "")
    features_str = "\n".join(features_tex)

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


def read_csv_text(csv_path: str) -> str:
    return Path(csv_path).read_text()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to a CSV projection matrix, such as CSV/projection_matrix.csv.",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label used for the matrix equation. Later adds \"eq:\" as a prefix)",
    )
    args = parser.parse_args()

    ans = latex_matrix(read_csv_text(args.csv_path), args.label)
    print(ans)
