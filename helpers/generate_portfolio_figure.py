import argparse


def env_title(env: str) -> str:
    return env.replace("_", " ").title()


def portfolio_figure(env: str, figure_prefix: str, is_svm: bool) -> str:
    caption = (
        f"Predicted algorithm for each instance in the {env_title(env)} environment."
        if is_svm
        else f"Best-performing algorithm for each instance in the {env_title(env)} environment."
    )
    label = (
        "res-" + env + "-svm-port"
        if is_svm
        else "res-" + env + "-port"
    )
    filename = (
        "6_svm_port"
        if is_svm
        else "5_port"
    )
    return """
\\begin{figure}
    \\centering
    \\includegraphics[width=\\figwidth\\linewidth]{""" + figure_prefix + filename + """.pdf}
    \\caption{""" + caption + """}
    \\label{fig:""" + label + """}
\\end{figure}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment key used in labels.")
    parser.add_argument("--figure-prefix", type=str, required=True, help="LaTeX path prefix for figure files.")
    parser.add_argument("--actual", action="store_true", help="Use actual portfolio figure instead of SVM predictions.")
    args = parser.parse_args()

    print(portfolio_figure(args.env, args.figure_prefix, not args.actual))
