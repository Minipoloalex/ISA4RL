import argparse


def env_title(env: str) -> str:
    return env.replace("_", " ").title()


def corr_figure(env: str, figure_prefix: str) -> str:
    small_caption = f"Correlation between metafeatures and algorithm performances for {env_title(env)}"
    return """
\\begin{figure}
    \\centering
    \\includegraphics[width=\\figwidth\\linewidth]{""" + figure_prefix + """1_corr_mf_perf.pdf}
    \\caption[""" + small_caption + """]{Pearson correlation coefficients between selected metafeatures and normalized algorithm performances for the """ \
        + env_title(env) + """ environment.}
    \\label{fig:res-""" + env + """-corr}
\\end{figure}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment key used in labels.")
    parser.add_argument("--figure-prefix", type=str, required=True, help="LaTeX path prefix for figure files.")
    args = parser.parse_args()

    print(corr_figure(args.env, args.figure_prefix))
