ENV = "merge"

SMALL_CAPTION = f"Correlation between metafeatures and algorithm performances for {ENV.capitalize()}"

ans = """
\\begin{figure}
    \\centering
    \\includegraphics[width=\\figwidth\\linewidth]{figures/results/single-envs/""" + ENV + """/1_corr_mf_perf.pdf}
    \\caption[""" + SMALL_CAPTION + """]{Pearson correlation coefficients between selected metafeatures and normalized algorithm performances for the """ \
        + ENV.capitalize() + """ environment.}
    \\label{fig:res-""" + ENV + """-corr}
\\end{figure}
"""
print(ans)
