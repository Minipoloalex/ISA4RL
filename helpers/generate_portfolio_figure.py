ENV = "merge"
IS_SVM = True

CAPTION = (
    f"Predicted algorithm for each instance in the {ENV.capitalize()} environment."
    if IS_SVM
    else f"Best-performing algorithm for each instance in the {ENV.capitalize()} environment."
)
LABEL = (
    "res-" + ENV + "-svm-port"
    if IS_SVM
    else "res-" + ENV + "-port"
)
FILENAME = (
    "6_svm_port"
    if IS_SVM
    else "5_port"
)

ans = """
\\begin{figure}
    \\centering
    \\includegraphics[width=\\figwidth\\linewidth]{figures/results/single-envs/""" + ENV + """/""" + FILENAME + """.pdf}
    \\caption{""" + CAPTION + """}
    \\label{fig:""" + LABEL + """}
\\end{figure}
"""
print(ans)
