import argparse
import csv
from pathlib import Path


def read_projection_features(csv_path: str) -> list[str]:
    with Path(csv_path).open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
    return header[1:]


def auto_nr_cols(item_count: int) -> list[int]:
    if item_count <= 3:
        return [item_count]
    rows = []
    remaining = item_count
    while remaining > 0:
        nr_cols = min(3, remaining)
        rows.append(nr_cols)
        remaining -= nr_cols
    return rows


def parse_nr_cols(raw_nr_cols: str) -> list[int]:
    return [int(value.strip()) for value in raw_nr_cols.split(",") if value.strip()]


def feature_distribution_figure(
    env: str,
    features: list[str],
    nr_cols: list[int],
    figure_prefix: str,
) -> str:
    if not all(feature.startswith("feature_") for feature in features):
        raise ValueError("All selected metafeatures must start with \"feature_\".")

    mfs = [feature[len("feature_"):] for feature in features]
    mfs = [f"2_{feature}" for feature in mfs]

    if len(mfs) != sum(nr_cols):
        raise ValueError("The number of selected metafeatures must equal the sum of --nr-cols.")
    if len(mfs) > 26:
        raise ValueError("Subfigure labels support at most 26 selected metafeatures.")

    label_prefix = f"fig:res-{env}"
    feature_index = 0
    figure_lines = [
        "\\begin{figure}",
        "    \\centering",
    ]

    for row_index, nr_col in enumerate(nr_cols):
        if row_index > 0:
            figure_lines.extend([
                "",
                "    \\medskip",
                "",
            ])

        width = f"{0.98 / nr_col:.3f}"

        for col_index in range(nr_col):
            mf = mfs[feature_index]
            subfigure_letter = chr(ord("a") + feature_index)
            feature_label = f"{label_prefix}-feat-{subfigure_letter}"
            feature_path = f"{figure_prefix}{mf}.pdf"
            feature_name = mf.removeprefix("2_").replace("_", "-")
            feature_caption = f"\\texttt{{{feature_name}}}\\label{{{feature_label}}}"

            figure_lines.extend([
                f"    \\subfloat[{feature_caption}]{{%",
                f"        \\includegraphics[width={width}\\textwidth]{{{feature_path}}}%",
                "    }",
            ])

            feature_index += 1

            if col_index < nr_col - 1:
                figure_lines.append("    \\hfill")

    figure_lines.extend([
        f"    \\caption{{Normalized feature distribution for the {len(mfs)} selected instance descriptors.}}",
        f"    \\label{{{label_prefix}-features}}",
        "\\end{figure}",
    ])
    return "\n".join(figure_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment key used in labels.")
    parser.add_argument("--projection-matrix", type=str, required=True, help="Path to CSV/projection_matrix.csv.")
    parser.add_argument("--figure-prefix", type=str, required=True, help="LaTeX path prefix for figure files.")
    parser.add_argument("--nr-cols", type=str, help="Comma-separated subfigure counts by row, such as 2,2,2.")
    args = parser.parse_args()

    selected_features = read_projection_features(args.projection_matrix)
    nr_cols = parse_nr_cols(args.nr_cols) if args.nr_cols else auto_nr_cols(len(selected_features))
    print(feature_distribution_figure(args.env, selected_features, nr_cols, args.figure_prefix))
