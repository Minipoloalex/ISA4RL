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
    file_path = f"train-configs/{file}"
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"{file}: {len(data)} configs")
    if file == "metadrive.json":
        for i, config in enumerate(data):
            # if i >= 225 and i <= 250:
            if i == 225:
                print(config["env_config"]["config"])
    total_size += len(data)

print(f"Total configs: {total_size}")

