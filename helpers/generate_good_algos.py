import argparse
import csv
from pathlib import Path

from generate_feature_distribution_figures import auto_nr_cols, parse_nr_cols


def env_title(env: str) -> str:
    return env.replace("_", " ").title()


def clean_algo(name: str) -> str:
    if name.startswith("algo_") and name.endswith("_mean_reward"):
        return name.replace("algo_", "").replace("_mean_reward", "").upper()
    return name


def read_algos_from_svm_table(csv_path: str) -> list[str]:
    with Path(csv_path).open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    return [clean_algo(row["Row"]) for row in rows if row["Row"].startswith("algo_")]


def good_algos_figure(
    env: str,
    algos: list[str],
    nr_cols: list[int],
    figure_prefix: str,
    svm_predictions: bool,
) -> str:
    if len(algos) != sum(nr_cols):
        raise ValueError("The number of algorithms must equal the sum of --nr-cols.")
    if len(algos) > 26:
        raise ValueError("Subfigure labels support at most 26 algorithms.")

    file_prefix = figure_prefix + ("4_svm_good_" if svm_predictions else "3_good_")
    label_prefix = f"fig:res-{env}-svm-good" if svm_predictions else f"fig:res-{env}-good"
    caption = (
        f"Predicted successful instances for each RL algorithm across the instance space for the {env_title(env)} environment."
        if svm_predictions
        else f"Successful instances for each RL algorithm across the instance space for the {env_title(env)} environment."
    )

    algo_index = 0
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
            algo = algos[algo_index]
            algo_key = algo.lower()
            subfigure_letter = chr(ord("a") + algo_index)
            algo_label = f"{label_prefix}-{algo_key}"
            algo_path = f"{file_prefix}{algo}.pdf"
            algo_caption = f"{algo}\\label{{{algo_label}}}"

            figure_lines.extend([
                f"    \\subfloat[{algo_caption}]{{%",
                f"        \\includegraphics[width={width}\\textwidth]{{{algo_path}}}%",
                "    }",
            ])

            algo_index += 1

            if col_index < nr_col - 1:
                figure_lines.append("    \\hfill")

    figure_lines.extend([
        "    \\caption{" + caption + "}",
        f"    \\label{{{label_prefix}}}",
        "\\end{figure}",
    ])
    return "\n".join(figure_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment key used in labels.")
    parser.add_argument("--svm-table", type=str, required=True, help="Path to CSV/svm_table.csv.")
    parser.add_argument("--figure-prefix", type=str, required=True, help="LaTeX path prefix for figure files.")
    parser.add_argument("--nr-cols", type=str, help="Comma-separated subfigure counts by row, such as 3 or 2,2.")
    parser.add_argument("--actual", action="store_true", help="Use actual good-instance figures instead of SVM predictions.")
    args = parser.parse_args()

    algos = read_algos_from_svm_table(args.svm_table)
    nr_cols = parse_nr_cols(args.nr_cols) if args.nr_cols else auto_nr_cols(len(algos))
    print(good_algos_figure(args.env, algos, nr_cols, args.figure_prefix, not args.actual))
