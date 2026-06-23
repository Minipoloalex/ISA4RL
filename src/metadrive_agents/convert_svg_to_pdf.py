import shutil
import subprocess
from pathlib import Path


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    """Convert an SVG file to a vector PDF using librsvg."""
    svg_path = svg_path.resolve()
    pdf_path = pdf_path.resolve()

    if svg_path.suffix.lower() != ".svg":
        raise ValueError(f"Expected an SVG input file, received: {svg_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF output file, received: {pdf_path}")
    if not svg_path.is_file():
        raise FileNotFoundError(f"SVG input file does not exist: {svg_path}")
    if svg_path.with_suffix(".pdf") == pdf_path:
        raise ValueError("The output name would overwrite the original image PDF")

    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required to convert SVG files to vector PDFs")

    subprocess.run(
        [converter, "--format", "pdf", "--output", str(pdf_path), str(svg_path)],
        check=True,
    )

    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"The PDF conversion did not produce a valid file: {pdf_path}")


def convert_generated_svgs(directory: Path) -> list[Path]:
    """Convert every generated image SVG without replacing the existing PDFs."""
    svg_paths = sorted(directory.glob("image_*.svg"))
    if not svg_paths:
        raise FileNotFoundError(f"No generated SVG images found in: {directory.resolve()}")

    pdf_paths = []
    for svg_path in svg_paths:
        pdf_path = svg_path.with_name(f"{svg_path.stem}_vector.pdf")
        convert_svg_to_pdf(svg_path, pdf_path)
        pdf_paths.append(pdf_path)
        print(f"Converted {svg_path.name} -> {pdf_path.name}")

    return pdf_paths


if __name__ == "__main__":
    convert_generated_svgs(Path(__file__).resolve().parent)
