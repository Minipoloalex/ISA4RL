ENV = "merge"

SMALL_CAPTION = (f"Observation types sources in the {ENV.capitalize()} environment.")

LABEL = f"res-{ENV}-sources"

ans = """
\\begin{figure}
    \\centering
    \\includegraphics[width=\\figwidth\\linewidth]{figures/results/single-envs/""" + ENV + """/7_sources.pdf}
    \\caption[""" + SMALL_CAPTION + """]{Observation types for each instance in the projection of the """ + \
        ENV.capitalize() + """ environment.}
    \\label{fig:""" + LABEL + """}
\\end{figure}
"""
print(ans)
