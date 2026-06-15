ALGOS = ["A2C", "DQN", "PPO"]
# ALGOS = ["A2C", "PPO", "SAC"]
ENV = "merge"
NR_COLS = [3]

SVM_PREDICTIONS = True

FILE_PREFIX = "figures/results/" + "single-envs/" + f"{ENV}/" + ("4_svm_good_" if SVM_PREDICTIONS else "3_good_")
LABEL_PREFIX = f"fig:res-{ENV}-svm-good" if SVM_PREDICTIONS else f"fig:res-{ENV}-good"
CAPTION = (
    f"Predicted successful instances for each RL algorithm across the instance space for the {ENV.capitalize()} environment."
    if SVM_PREDICTIONS
    else f"Successful instances for each RL algorithm across the instance space for the {ENV.capitalize()} environment."
)

assert len(ALGOS) == sum(NR_COLS)
assert len(ALGOS) <= 26

algo_index = 0
figure_lines = [
    "\\begin{figure}",
    "    \\centering",
]

for row_index, nr_cols in enumerate(NR_COLS):
    if row_index > 0:
        figure_lines.extend([
            "",
            "    \\medskip",
            "",
        ])

    width = f"{0.98 / nr_cols:.3f}"

    for col_index in range(nr_cols):
        algo = ALGOS[algo_index]
        algo_key = algo.lower()
        subfigure_letter = chr(ord("a") + algo_index)
        algo_label = f"{LABEL_PREFIX}-{algo_key}"
        algo_path = f"{FILE_PREFIX}{algo_key}.pdf"
        algo_caption = f"{algo}\\label{{{algo_label}}}"

        figure_lines.extend([
            f"    \\subfloat[{algo_caption}]{{%",
            f"        \\includegraphics[width={width}\\textwidth]{{{algo_path}}}%",
            "    }",
        ])

        algo_index += 1

        if col_index < nr_cols - 1:
            figure_lines.append("    \\hfill")

figure_lines.extend([
    "    \\caption{" + CAPTION + "}",
    f"    \\label{{{LABEL_PREFIX}}}",
    "\\end{figure}",
])

latex_string = "\n".join(figure_lines)

print(latex_string)
