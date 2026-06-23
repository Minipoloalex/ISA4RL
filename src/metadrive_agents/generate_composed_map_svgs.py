import html
import logging
import math
from pathlib import Path

import numpy as np

from convert_svg_to_pdf import convert_svg_to_pdf
from metadrive import MetaDriveEnv
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.extension_lane import ExtendingLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.constants import PGLineColor, PGLineType


MAP_NAMES = ("SC", "rORY", "TXT")
SVG_SIZE = 600
PATH_PRECISION = 3


def format_point(point: np.ndarray) -> str:
    return f"{float(point[0]):.{PATH_PRECISION}f} {-float(point[1]):.{PATH_PRECISION}f}"


def points_to_path(points: np.ndarray, close_path: bool) -> str:
    command = f"M {format_point(points[0])}"
    if len(points) > 1:
        command += " L " + " ".join(format_point(point) for point in points[1:])
    if close_path:
        command += " Z"
    return command


def sample_lane_boundary(lane, side_index: int) -> np.ndarray:
    if isinstance(lane, CircularLane):
        sample_count = max(3, math.ceil(abs(float(lane.angle)) / math.radians(4)) + 1)
    elif isinstance(lane, ExtendingLane):
        sample_count = 24
    elif isinstance(lane, StraightLane):
        sample_count = 2
    else:
        sample_count = max(3, math.ceil(float(lane.length) / 2) + 1)

    longitudinal_positions = np.linspace(0, lane.length, sample_count)
    points = []
    for longitudinal in longitudinal_positions:
        lane_width = lane.width_at(float(longitudinal))
        lateral = -lane.width / 2 if side_index == 0 else lane_width - lane.width / 2
        points.append(lane.position(float(longitudinal), lateral))
    return np.asarray(points)


def canonical_path_key(points: np.ndarray) -> tuple:
    rounded_points = tuple((round(float(point[0]), 2), round(float(point[1]), 2)) for point in points)
    reversed_points = tuple(reversed(rounded_points))
    return min(rounded_points, reversed_points)


def line_style(line_type, line_color) -> tuple[str, int] | None:
    if line_type == PGLineType.NONE:
        return None
    if line_type == PGLineType.BROKEN:
        return "divider", 1
    if line_type == PGLineType.SIDE:
        return "edge", 0
    if line_type == PGLineType.CONTINUOUS:
        if tuple(line_color) == tuple(PGLineColor.YELLOW):
            return "center-line", 2
        return "edge", 0
    raise ValueError(f"Unsupported MetaDrive line type: {line_type}")


def get_map_lanes(env: MetaDriveEnv) -> list:
    lanes = []
    seen_lane_ids = set()
    for end_nodes in env.current_map.road_network.graph.values():
        for road_lanes in end_nodes.values():
            for lane in road_lanes:
                lane_id = id(lane)
                if lane_id not in seen_lane_ids:
                    seen_lane_ids.add(lane_id)
                    lanes.append(lane)
    if not lanes:
        raise RuntimeError("The generated MetaDrive map contains no lanes")
    return lanes


def build_svg(map_name: str, lanes: list) -> str:
    surface_paths = []
    all_points = []
    line_paths = {}

    for lane in lanes:
        polygon = np.asarray(lane.polygon)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError(f"Invalid lane polygon for map {map_name}: {polygon.shape}")
        surface_paths.append(points_to_path(polygon, close_path=True))
        all_points.append(polygon)

        if len(lane.line_types) != 2 or len(lane.line_colors) != 2:
            raise ValueError("Each lane must define exactly two line types and colors")
        for side_index in range(2):
            style = line_style(lane.line_types[side_index], lane.line_colors[side_index])
            if style is None:
                continue
            points = sample_lane_boundary(lane, side_index)
            all_points.append(points)
            key = canonical_path_key(points)
            previous = line_paths.get(key)
            if previous is None or style[1] > previous[0]:
                line_paths[key] = (style[1], style[0], points)

    combined_points = np.concatenate(all_points)
    svg_points = np.column_stack((combined_points[:, 0], -combined_points[:, 1]))
    minimum = svg_points.min(axis=0)
    maximum = svg_points.max(axis=0)
    dimensions = maximum - minimum
    padding = max(float(dimensions.max()) * 0.025, 3.5)
    view_x = float(minimum[0] - padding)
    view_y = float(minimum[1] - padding)
    view_width = float(dimensions[0] + 2 * padding)
    view_height = float(dimensions[1] + 2 * padding)

    surface_markup = "\n    ".join(f"<path d=\"{path}\"/>" for path in surface_paths)
    sorted_lines = sorted(line_paths.values(), key=lambda item: item[0])
    line_markup = "\n  ".join(
        f"<path class=\"{class_name}\" d=\"{points_to_path(points, close_path=False)}\"/>"
        for _, class_name, points in sorted_lines
    )

    escaped_name = html.escape(map_name)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{SVG_SIZE}"
     height="{SVG_SIZE}"
     viewBox="{view_x:.3f} {view_y:.3f} {view_width:.3f} {view_height:.3f}"
     role="img"
     aria-labelledby="title description">
  <title id="title">MetaDrive composed map {escaped_name}</title>
  <desc id="description">
    Seed-zero top-down vector rendering of the composed MetaDrive map {escaped_name}.
  </desc>
  <defs>
    <style>
      .surfaces {{ fill: #d9d9d9; stroke: none; }}
      .edge {{ fill: none; stroke: #92979b; stroke-width: 0.32; stroke-linecap: butt; stroke-linejoin: round; }}
      .divider {{ fill: none; stroke: #8d9296; stroke-width: 0.28; stroke-dasharray: 1.8 1.8; stroke-linecap: butt; stroke-linejoin: round; }}
      .center-line {{ fill: none; stroke: #4f5356; stroke-width: 0.38; stroke-linecap: butt; stroke-linejoin: round; }}
    </style>
  </defs>
  <g class="surfaces">
    {surface_markup}
  </g>
  {line_markup}
</svg>
"""


def generate_composed_map_graphic(map_name: str, output_directory: Path) -> tuple[Path, Path]:
    env = MetaDriveEnv(dict(num_scenarios=1, map=map_name, log_level=logging.WARNING))
    try:
        env.reset(seed=0)
        lanes = get_map_lanes(env)
        svg = build_svg(map_name, lanes)
    finally:
        env.close()

    svg_path = output_directory / f"image_{map_name}.svg"
    pdf_path = output_directory / f"image_{map_name}_vector.pdf"
    svg_path.write_text(svg, encoding="utf-8")
    convert_svg_to_pdf(svg_path, pdf_path)
    print(f"Generated {svg_path.name} and {pdf_path.name}")
    return svg_path, pdf_path


if __name__ == "__main__":
    output_directory = Path(__file__).resolve().parent / "map_visualizations"
    if not output_directory.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_directory}")
    for map_name in MAP_NAMES:
        generate_composed_map_graphic(map_name, output_directory)
