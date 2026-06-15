import argparse
import re
from pathlib import Path

from csv_to_latex_matrix import latex_matrix, read_csv_text
from csv_to_latex_table import latex_table
from generate_corr_fig import corr_figure
from generate_feature_distribution_figures import (
    auto_nr_cols,
    feature_distribution_figure,
    read_projection_features,
)
from generate_good_algos import good_algos_figure, read_algos_from_svm_table
from generate_portfolio_figure import portfolio_figure
from generate_sources_fig import sources_figure


HELPERS_DIR = Path(__file__).resolve().parent


def infer_env(run_dir: Path) -> str:
    match = re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(?P<env>.+?)(?:_all)?$", run_dir.name)
    if match is None:
        raise ValueError(
            "Could not infer the environment from the ISA run folder name. "
            "Pass --env explicitly."
        )
    return match.group("env")


def default_output_dir(run_dir: Path) -> Path:
    return HELPERS_DIR / "latex" / run_dir.name


def latex_figure_prefix(env: str) -> str:
    return f"figures/results/single-envs/{env}/"


def write_text(output_dir: Path, filename: str, text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text(text.strip() + "\n")


def required_file(run_dir: Path, relative_path: str) -> Path:
    path = run_dir / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def validate_pdf_outputs(run_dir: Path, selected_features: list[str], selected_algos: list[str]) -> None:
    expected_pdfs = [
        "1_corr_mf_perf.pdf",
        "5_port.pdf",
        "6_svm_port.pdf",
        "7_sources.pdf",
    ]
    for feature in selected_features:
        feature_name = feature.removeprefix("feature_")
        expected_pdfs.append(f"2_{feature_name}.pdf")
    expected_pdfs.extend(
        f"{prefix}{algo}.pdf"
        for algo in selected_algos
        for prefix in ["3_good_", "4_svm_good_"]
    )

    missing_pdfs = [
        str(run_dir / "PDF" / filename)
        for filename in expected_pdfs
        if not (run_dir / "PDF" / filename).exists()
    ]
    if missing_pdfs:
        raise FileNotFoundError("Missing expected PDF output(s):\n" + "\n".join(missing_pdfs))


def generate_everything(run_dir: Path, env: str, output_dir: Path, figure_prefix: str) -> list[Path]:
    projection_matrix = required_file(run_dir, "CSV/projection_matrix.csv")
    svm_table = required_file(run_dir, "CSV/svm_table.csv")

    selected_features = read_projection_features(str(projection_matrix))
    selected_algos = read_algos_from_svm_table(str(svm_table))
    validate_pdf_outputs(run_dir, selected_features, selected_algos)
    env_name = env.replace("_", " ").title()

    artifacts = {
        "projection_matrix.tex": latex_matrix(read_csv_text(str(projection_matrix)), f"res-{env}-mat"),
        "performance_table.tex": latex_table(
            str(svm_table),
            f"tab:res-{env}",
            f"Algorithm Performance Comparison for the {env_name} environment.",
        ),
        "correlation_figure.tex": corr_figure(env, figure_prefix),
        "feature_distribution_figures.tex": feature_distribution_figure(
            env,
            selected_features,
            auto_nr_cols(len(selected_features)),
            figure_prefix,
        ),
        "svm_good_algos_figure.tex": good_algos_figure(
            env,
            selected_algos,
            auto_nr_cols(len(selected_algos)),
            figure_prefix,
            True,
        ),
        "good_algos_figure.tex": good_algos_figure(
            env,
            selected_algos,
            auto_nr_cols(len(selected_algos)),
            figure_prefix,
            False,
        ),
        "svm_portfolio_figure.tex": portfolio_figure(env, figure_prefix, True),
        "portfolio_figure.tex": portfolio_figure(env, figure_prefix, False),
        "sources_figure.tex": sources_figure(env, figure_prefix),
    }

    written_paths = []
    for filename, text in artifacts.items():
        write_text(output_dir, filename, text)
        written_paths.append(output_dir / filename)

    write_text(output_dir, "all.tex", "\n\n".join(artifacts.values()))
    written_paths.append(output_dir / "all.tex")
    return written_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "isa_run_dir",
        type=Path,
        help="Path to an ISA run folder, such as ../results/isa/2026-06-15_09-55-19_merge_all.",
    )
    parser.add_argument("--env", type=str, help="Environment key used in labels.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where generated .tex snippets should be saved.",
    )
    parser.add_argument(
        "--figure-prefix",
        type=str,
        help="LaTeX path prefix used inside \\includegraphics commands.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.isa_run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    env = args.env if args.env is not None else infer_env(run_dir)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_output_dir(run_dir)
    figure_prefix = args.figure_prefix if args.figure_prefix is not None else latex_figure_prefix(env)

    written_paths = generate_everything(run_dir, env, output_dir, figure_prefix)
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
