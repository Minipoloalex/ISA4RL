from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OBSERVATION_SOURCE_NAMES = {
    "Kinematics": "Kinematics",
    "GrayscaleObservation": "Image",
    "TimeToCollision": "TTC",
    "KinematicsGoal": "Kinematics goal",
    "ExitObservation": "Exit observation",
    "AttributesObservation": "Attributes",
    "OccupancyGrid": "Occupancy grid",
    "state_observation": "State observation",
    "image_observation": "Image observation",
}


class ReplaceSourceWithObservationError(RuntimeError):
    """Raised when an ISA CSV source cannot be replaced with observation labels."""


def read_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def instance_config_path(results_dir: Path, environment: str, instance_id: str) -> Path:
    return results_dir / environment / instance_id / "instance_config.json"


def observation_type_from_config(config: dict, config_path: Path) -> str:
    obs_config = config.get("obs_config")
    if isinstance(obs_config, dict) and "type" in obs_config:
        return obs_config["type"]

    env_config = config.get("env_config")
    if isinstance(env_config, dict):
        environment_config = env_config.get("config")
        if isinstance(environment_config, dict) and "image_observation" in environment_config:
            if environment_config["image_observation"] is True:
                return "image_observation"
            if environment_config["image_observation"] is False:
                return "state_observation"

    raise ReplaceSourceWithObservationError(
        f"Could not find an observation type in instance config: '{config_path}'."
    )


def observation_source_name(observation_type: str) -> str:
    if observation_type not in OBSERVATION_SOURCE_NAMES:
        raise ReplaceSourceWithObservationError(
            f"Observation type is not mapped to a source name: '{observation_type}'. "
            "Edit OBSERVATION_SOURCE_NAMES in this script."
        )

    return OBSERVATION_SOURCE_NAMES[observation_type]


def source_name_for_instance(results_dir: Path, environment: str, instance_id: str) -> str:
    config_path = instance_config_path(results_dir, environment, instance_id)
    if not config_path.is_file():
        raise ReplaceSourceWithObservationError(f"Instance config not found: '{config_path}'.")

    observation_type = observation_type_from_config(read_json(config_path), config_path)
    return observation_source_name(observation_type)


def replace_source_with_observation(
    csv_path: Path,
    output_path: Path,
    results_dir: Path,
    environment: str,
) -> None:
    csv_path = csv_path.expanduser()
    output_path = output_path.expanduser()
    results_dir = results_dir.expanduser()

    if not csv_path.is_file():
        raise ReplaceSourceWithObservationError(f"CSV file not found: '{csv_path}'.")
    if not results_dir.is_dir():
        raise ReplaceSourceWithObservationError(f"Results directory not found: '{results_dir}'.")

    dataframe = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if "instances" not in dataframe.columns:
        raise ReplaceSourceWithObservationError(
            f"CSV file does not contain an 'instances' column: '{csv_path}'."
        )
    if "source" not in dataframe.columns:
        raise ReplaceSourceWithObservationError(
            f"CSV file does not contain a 'source' column: '{csv_path}'."
        )
    if dataframe["instances"].str.strip().eq("").any():
        raise ReplaceSourceWithObservationError(f"CSV file contains empty instances: '{csv_path}'.")

    output_dataframe = dataframe.copy()
    output_dataframe["source"] = [
        source_name_for_instance(results_dir, environment, instance_id)
        for instance_id in output_dataframe["instances"]
    ]
    output_dataframe.to_csv(output_path, index=False)

    print(f"Wrote {len(output_dataframe)} rows from '{csv_path}' to '{output_path}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace the source column in an ISA CSV with labels derived from each "
            "instance observation type."
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
        required=True,
        help="Environment folder name inside the results directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        replace_source_with_observation(
            args.csv,
            args.output,
            args.results_dir,
            args.environment,
        )
    except ReplaceSourceWithObservationError as err:
        print(f"[replace_source_with_observation] {err}")
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
