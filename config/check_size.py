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

env_files = [f"{env}.json" for env in ENVS]

total_size = 0
for file in env_files:
    file_path = f"eval-configs/{file}"
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"{file}: {len(data)} configs")
    total_size += len(data)

print(f"Total configs: {total_size}")

