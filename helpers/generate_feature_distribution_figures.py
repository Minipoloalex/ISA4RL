ENV = "merge"
MFS = "feature_lanes_count,feature_random_collision_rate,feature_baseline_obs_volatility_mean,feature_baseline_other_veh_speed_delta_mean,feature_baseline_other_veh_heading_delta_mean,feature_action_state_discontinuity_mean"
NR_COLS = [2,2,2]

FILE_PREFIX = "figures/results/" + "single-envs/" + f"{ENV}/"
LABEL_PREFIX = f"fig:res-{ENV}"

raw_mfs = MFS.split(",")
assert all(mf.startswith("feature_") for mf in raw_mfs)

MFS = [mf[len("feature_"):] for mf in raw_mfs]
MFS = [f"2_{mf}" for mf in MFS]

assert len(MFS) == sum(NR_COLS)
assert len(MFS) <= 26

feature_index = 0
figure_lines = [
    "\\begin{figure}",
    "    \\centering",
]
caption_items = []

for row_index, nr_cols in enumerate(NR_COLS):
    if row_index > 0:
        figure_lines.extend([
            "",
            "    \\medskip",
            "",
        ])

    width = f"{0.98 / nr_cols:.3f}"

    for col_index in range(nr_cols):
        mf = MFS[feature_index]
        subfigure_letter = chr(ord("a") + feature_index)
        feature_label = f"{LABEL_PREFIX}-feat-{subfigure_letter}"
        feature_path = f"{FILE_PREFIX}{mf}.pdf"
        feature_name = mf.removeprefix("2_").replace("_", "-")
        feature_caption = f"\\texttt{{{feature_name}}}\\label{{{feature_label}}}"

        figure_lines.extend([
            f"    \\subfloat[{feature_caption}]{{%",
            f"        \\includegraphics[width={width}\\textwidth]{{{feature_path}}}%",
            "    }",
        ])

        caption_items.append(f"({subfigure_letter}) \\texttt{{{feature_name}}}")
        feature_index += 1

        if col_index < nr_cols - 1:
            figure_lines.append("    \\hfill")

figure_lines.extend([
    f"    \\caption{{Normalized feature distribution for the {len(MFS)} selected instance descriptors.}}",
    f"    \\label{{{LABEL_PREFIX}-features}}",
    "\\end{figure}",
])

latex_string = "\n".join(figure_lines)

print(latex_string)
