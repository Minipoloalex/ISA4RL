from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainRun:
    source_instance_dir: Path
    run_dir: Path


@dataclass(frozen=True)
class InstanceGroup:
    keeper_dir: Path
    source_dirs: list[Path]
    train_runs: list[TrainRun]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a cleaned result folder with one folder per unique instance "
            "and only train runs that contain the evaluation metadata JSON."
        )
    )
    parser.add_argument(
        "environment",
        nargs="?",
        default="parking",
        help="Environment results folder name, for example parking, exit, or merge.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root folder that contains environment result folders.",
    )
    parser.add_argument(
        "--output-environment",
        default=None,
        help="Output environment folder name. Defaults to <environment>-new.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing the output folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output folder first if it already exists.",
    )
    parser.add_argument(
        "--evaluation-metadata-file",
        default="training_metadata.json",
        help=(
            "File that must exist inside train/<run-id> for that run folder to be copied."
        ),
    )
    return parser.parse_args()


def canonical_json_hash(json_path: Path) -> str:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_instance_dirs(environment_dir: Path) -> list[Path]:
    instance_dirs: list[Path] = []
    for child in sorted(environment_dir.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue

        instance_config_path = child / "instance_config.json"
        if instance_config_path.is_file():
            instance_dirs.append(child)

    return instance_dirs


def contains_evaluation_metadata(run_dir: Path, evaluation_metadata_file: str) -> bool:
    return (run_dir / evaluation_metadata_file).is_file()


def find_train_runs(instance_dir: Path, evaluation_metadata_file: str) -> list[TrainRun]:
    train_dir = instance_dir / "train"
    if not train_dir.is_dir():
        return []

    train_runs: list[TrainRun] = []
    for run_dir in sorted(train_dir.iterdir(), key=lambda path: path.name):
        if not run_dir.is_dir():
            continue

        if contains_evaluation_metadata(run_dir, evaluation_metadata_file):
            train_runs.append(TrainRun(source_instance_dir=instance_dir, run_dir=run_dir))

    return train_runs


def group_instances(
    instance_dirs: list[Path], evaluation_metadata_file: str
) -> list[InstanceGroup]:
    groups_by_hash: dict[str, list[Path]] = {}
    ordered_hashes: list[str] = []

    for instance_dir in instance_dirs:
        instance_hash = canonical_json_hash(instance_dir / "instance_config.json")
        if instance_hash not in groups_by_hash:
            groups_by_hash[instance_hash] = []
            ordered_hashes.append(instance_hash)
        groups_by_hash[instance_hash].append(instance_dir)

    groups: list[InstanceGroup] = []
    for instance_hash in ordered_hashes:
        source_dirs = groups_by_hash[instance_hash]
        train_runs: list[TrainRun] = []
        for source_dir in source_dirs:
            train_runs.extend(find_train_runs(source_dir, evaluation_metadata_file))

        groups.append(
            InstanceGroup(
                keeper_dir=source_dirs[0],
                source_dirs=source_dirs,
                train_runs=train_runs,
            )
        )

    return groups


def copy_instance_metadata(source_instance_dir: Path, destination_instance_dir: Path) -> None:
    destination_instance_dir.mkdir(parents=True)
    for source_path in sorted(source_instance_dir.iterdir(), key=lambda path: path.name):
        if source_path.name == "train":
            continue

        destination_path = destination_instance_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)


def get_destination_run_dir(train_run: TrainRun, destination_train_dir: Path) -> Path:
    destination_run_dir = destination_train_dir / train_run.run_dir.name
    if not destination_run_dir.exists():
        return destination_run_dir

    source_id = train_run.source_instance_dir.name
    destination_run_dir = destination_train_dir / f"{train_run.run_dir.name}__from_{source_id}"
    if not destination_run_dir.exists():
        return destination_run_dir

    raise FileExistsError(f"Conflicting train run destination: {destination_run_dir}")


def write_manifest(output_dir: Path, groups: list[InstanceGroup]) -> None:
    manifest = {
        "groups": [
            {
                "kept_instance_id": group.keeper_dir.name,
                "source_instance_ids": [source_dir.name for source_dir in group.source_dirs],
                "copied_train_runs": [
                    {
                        "source_instance_id": train_run.source_instance_dir.name,
                        "train_run_id": train_run.run_dir.name,
                    }
                    for train_run in group.train_runs
                ],
            }
            for group in groups
        ]
    }

    manifest_path = output_dir / "unclog_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


def create_clean_results(output_dir: Path, groups: list[InstanceGroup]) -> None:
    output_dir.mkdir(parents=True)

    for group in groups:
        destination_instance_dir = output_dir / group.keeper_dir.name
        copy_instance_metadata(group.keeper_dir, destination_instance_dir)

        destination_train_dir = destination_instance_dir / "train"
        destination_train_dir.mkdir()

        for train_run in group.train_runs:
            destination_run_dir = get_destination_run_dir(train_run, destination_train_dir)
            shutil.copytree(train_run.run_dir, destination_run_dir)

    write_manifest(output_dir, groups)


def print_summary(
    input_dir: Path,
    output_dir: Path,
    groups: list[InstanceGroup],
    evaluation_metadata_file: str,
) -> None:
    source_instance_count = sum(len(group.source_dirs) for group in groups)
    duplicate_instance_count = sum(len(group.source_dirs) - 1 for group in groups)
    train_run_count = sum(len(group.train_runs) for group in groups)
    groups_with_evaluation_count = sum(1 for group in groups if group.train_runs)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Required evaluation metadata file: {evaluation_metadata_file}")
    print(f"Source instance folders: {source_instance_count}")
    print(f"Unique instances: {len(groups)}")
    print(f"Duplicate folders collapsed: {duplicate_instance_count}")
    print(f"Unique instances with copied evaluation runs: {groups_with_evaluation_count}")
    print(f"Training run folders copied: {train_run_count}")


def main() -> None:
    args = parse_args()

    input_dir = args.results_root / args.environment
    output_environment = args.output_environment
    if output_environment is None:
        output_environment = f"{args.environment}-new"
    output_dir = args.results_root / output_environment

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input environment folder does not exist: {input_dir}")

    instance_dirs = get_instance_dirs(input_dir)
    groups = group_instances(instance_dirs, args.evaluation_metadata_file)
    print_summary(input_dir, output_dir, groups, args.evaluation_metadata_file)

    if args.dry_run:
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output folder already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    create_clean_results(output_dir, groups)


if __name__ == "__main__":
    main()
