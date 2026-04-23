import json
from typing import List

HIGHWAY_ENVS: List[str] = [
    "highway",
    "roundabout",
    "merge",
    "exit",
    "racetrack",
    "parking",
    "lane-keeping",
]
METADRIVE_ENVS: List[str] = ["metadrive"]
ENVS = HIGHWAY_ENVS + METADRIVE_ENVS

env_files = [f"{env}-configs.json" for env in ENVS]

for file in env_files:
    file_path = f"env-configs/{file}"
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"{file}: {len(data)} configs")

