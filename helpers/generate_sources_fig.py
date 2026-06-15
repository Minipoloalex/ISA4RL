import argparse


def env_title(env: str) -> str:
    return env.replace("_", " ").title()


def sources_figure(env: str, figure_prefix: str) -> str:
    small_caption = f"Observation types sources in the {env_title(env)} environment."
    label = f"res-{env}-sources"
    return """
\\begin{figure}
    \\centering
    \\includegraphics[width=\\figwidth\\linewidth]{""" + figure_prefix + """7_sources.pdf}
    \\caption[""" + small_caption + """]{Observation types for each instance in the projection of the """ + \
        env_title(env) + """ environment.}
    \\label{fig:""" + label + """}
\\end{figure}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment key used in labels.")
    parser.add_argument("--figure-prefix", type=str, required=True, help="LaTeX path prefix for figure files.")
    args = parser.parse_args()

    print(sources_figure(args.env, args.figure_prefix))
