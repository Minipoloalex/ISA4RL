from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MAP_NAMES_MAPPING = {
    "rORY": "Merge + Roundabout + Exit",
    "SC": "Straight + Curve",
    "TXT": "3 Intersections",
    # "Intersection / Roundabout"
}

class ReplaceMetadriveSourceWithMapError(RuntimeError):
    """Raised when an ISA CSV source cannot be replaced with MetaDrive maps."""


def read_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def instance_config_path(results_dir: Path, environment: str, instance_id: str) -> Path:
    return results_dir / environment / instance_id / "instance_config.json"


def metadrive_map_from_config(config: dict, config_path: Path) -> str:
    env_config = config.get("env_config")
    if not isinstance(env_config, dict):
        raise ReplaceMetadriveSourceWithMapError(
            f"Could not find 'env_config' in instance config: '{config_path}'."
        )

    environment_config = env_config.get("config")
    if not isinstance(environment_config, dict):
        raise ReplaceMetadriveSourceWithMapError(
            f"Could not find 'env_config.config' in instance config: '{config_path}'."
        )

    map_config = environment_config.get("map_config")
    if not isinstance(map_config, dict):
        raise ReplaceMetadriveSourceWithMapError(
            f"Could not find 'env_config.config.map_config' in instance config: "
            f"'{config_path}'."
        )

    map_name = map_config.get("config")
    if not isinstance(map_name, str) or not map_name.strip():
        map_name = "Intersection / Roundabout"
        # raise ReplaceMetadriveSourceWithMapError(
        #     f"Could not find a non-empty string in "
        #     f"'env_config.config.map_config.config': '{config_path}'."
        # )
    map_name = MAP_NAMES_MAPPING.get(map_name, map_name)
    return map_name


def map_source_for_instance(results_dir: Path, environment: str, instance_id: str) -> str:
    config_path = instance_config_path(results_dir, environment, instance_id)
    if not config_path.is_file():
        raise ReplaceMetadriveSourceWithMapError(
            f"Instance config not found: '{config_path}'."
        )

    return metadrive_map_from_config(read_json(config_path), config_path)


def replace_metadrive_source_with_map(
    csv_path: Path,
    output_path: Path,
    results_dir: Path,
    environment: str,
) -> None:
    csv_path = csv_path.expanduser()
    output_path = output_path.expanduser()
    results_dir = results_dir.expanduser()

    if not csv_path.is_file():
        raise ReplaceMetadriveSourceWithMapError(f"CSV file not found: '{csv_path}'.")
    if not results_dir.is_dir():
        raise ReplaceMetadriveSourceWithMapError(
            f"Results directory not found: '{results_dir}'."
        )

    dataframe = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if "instances" not in dataframe.columns:
        raise ReplaceMetadriveSourceWithMapError(
            f"CSV file does not contain an 'instances' column: '{csv_path}'."
        )
    if "source" not in dataframe.columns:
        raise ReplaceMetadriveSourceWithMapError(
            f"CSV file does not contain a 'source' column: '{csv_path}'."
        )
    if dataframe["instances"].str.strip().eq("").any():
        raise ReplaceMetadriveSourceWithMapError(
            f"CSV file contains empty instances: '{csv_path}'."
        )

    output_dataframe = dataframe.copy()
    output_dataframe["source"] = [
        map_source_for_instance(results_dir, environment, instance_id)
        for instance_id in output_dataframe["instances"]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(output_path, index=False)

    print(f"Wrote {len(output_dataframe)} rows from '{csv_path}' to '{output_path}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace the source column in a MetaDrive ISA CSV with the map of "
            "each instance."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input ISA CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output ISA CSV path.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Results root containing <environment>/<instance>/instance_config.json folders.",
    )
    parser.add_argument(
        "--environment",
        default="metadrive",
        help="Environment folder name inside the results directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        replace_metadrive_source_with_map(
            args.csv,
            args.output,
            args.results_dir,
            args.environment,
        )
    except ReplaceMetadriveSourceWithMapError as err:
        print(f"[replace_metadrive_source_with_map] {err}")
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
