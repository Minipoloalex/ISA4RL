import argparse
import csv
from pathlib import Path


def clean_name(name):
    if name.startswith("algo_") and name.endswith("_mean_reward"):
        return name.replace("algo_", "").replace("_mean_reward", "").upper()
    return name


def read_table_rows(csv_path: str) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def sort_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    order = ["A2C", "DQN", "PPO", "SAC", "Selector", "Oracle"]
    order_by_name = {name: index for index, name in enumerate(order)}
    return sorted(rows, key=lambda row: order_by_name.get(row["Row"], len(order_by_name)))


def latex_table(csv_path: str, label: str, caption: str) -> str:
    rows = read_table_rows(csv_path)
    for row in rows:
        row["Row"] = clean_name(row["Row"])

    rows = sort_rows(rows)
    max_prob = max(float(row["Probability_of_good"]) for row in rows if row["Row"] != "Oracle")

    latex_lines = [
        "\\begin{table}",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{cccc}",
        "\\toprule",
        "\\textbf{Algorithms} & \\textbf{\\ Average performance\\ } & \\textbf{\\ Std performance\\ } & \\textbf{Probability of good} \\\\ \\midrule",
    ]

    for row in rows:
        algo = row["Row"]
        avg_perf = row["Avg_Perf_all_instances"]
        std_perf = row["Std_Perf_all_instances"]
        avg = f"{float(avg_perf):.3f}"
        std = f"{float(std_perf):.3f}"
        prob = float(row["Probability_of_good"])
        prob_str = f"{prob:.3f}"

        if algo == "Selector":
            algo = f"\\textbf{{{algo}}}"

        if prob == max_prob and row["Row"] != "Oracle":
            prob_str = f"\\textbf{{{prob_str}}}"

        latex_lines.append(f"{algo} & {avg} & {std} & {prob_str} \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(latex_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str, help="Path to CSV/svm_table.csv.")
    parser.add_argument("--label", type=str, required=True, help="LaTeX table label.")
    parser.add_argument("--caption", type=str, required=True, help="LaTeX table caption.")
    args = parser.parse_args()

    print(latex_table(args.csv_path, args.label, args.caption))
