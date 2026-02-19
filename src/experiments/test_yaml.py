from pathlib import Path
from typing import Any, Dict

import yaml

def parse_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    return data or {}

if __name__ == "__main__":
    algo_config = parse_yaml("config/algo-configurations.yaml")
    env_config = parse_yaml("config/env-configurations.yaml")

    test = {
        "algo": "ppo",
        "env" : "highway",
    }

    print(algo_config[test["algo"]])
    print(env_config[test["env"]])
