from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TRAINING_METADATA_FILE = "training_metadata.json"


@dataclass(frozen=True)
class CleanupPlan:
    environment_dir: Path
    kept_instance_dirs: list[Path]
    removable_instance_dirs: list[Path]
    metadata_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove instance result folders for a specific environment when they do "
            "not contain a completed training metadata file."
        )
    )
    parser.add_argument(
        "environment",
        help="Environment results folder name, for example parking, exit, or merge.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root folder that contains environment result folders.",
    )
    parser.add_argument(
        "--training-metadata-file",
        default=DEFAULT_TRAINING_METADATA_FILE,
        help="File that must exist inside train/<run-id> to keep an instance folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting anything.",
    )
    return parser.parse_args()


def has_training_metadata(instance_dir: Path, training_metadata_file: str) -> bool:
    train_dir = instance_dir / "train"
    if not train_dir.is_dir():
        return False

    for run_dir in sorted(train_dir.iterdir(), key=lambda path: path.name):
        if not run_dir.is_dir():
            continue

        metadata_path = run_dir / training_metadata_file
        if metadata_path.is_file():
            return True

    return False


def build_cleanup_plan(
    environment_dir: Path, training_metadata_file: str
) -> CleanupPlan:
    kept_instance_dirs: list[Path] = []
    removable_instance_dirs: list[Path] = []

    for instance_dir in sorted(environment_dir.iterdir(), key=lambda path: path.name):
        if not instance_dir.is_dir():
            continue

        if has_training_metadata(instance_dir, training_metadata_file):
            kept_instance_dirs.append(instance_dir)
        else:
            removable_instance_dirs.append(instance_dir)

    return CleanupPlan(
        environment_dir=environment_dir,
        kept_instance_dirs=kept_instance_dirs,
        removable_instance_dirs=removable_instance_dirs,
        metadata_file=training_metadata_file,
    )


def print_plan(plan: CleanupPlan, dry_run: bool) -> None:
    action = "Would remove" if dry_run else "Removing"

    print(f"Environment folder: {plan.environment_dir}")
    print(f"Required training metadata file: {plan.metadata_file}")
    print(f"Instance folders kept: {len(plan.kept_instance_dirs)}")
    print(f"Instance folders selected for removal: {len(plan.removable_instance_dirs)}")

    for instance_dir in plan.removable_instance_dirs:
        print(f"{action}: {instance_dir}")


def remove_instance_dirs(instance_dirs: list[Path]) -> None:
    for instance_dir in instance_dirs:
        shutil.rmtree(instance_dir)


def main() -> None:
    args = parse_args()
    environment_dir = args.results_root / args.environment

    if not environment_dir.is_dir():
        raise NotADirectoryError(
            f"Environment results folder does not exist: {environment_dir}"
        )

    plan = build_cleanup_plan(environment_dir, args.training_metadata_file)
    print_plan(plan, args.dry_run)

    if args.dry_run:
        return

    remove_instance_dirs(plan.removable_instance_dirs)


if __name__ == "__main__":
    main()
