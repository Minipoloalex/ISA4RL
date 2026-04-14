from typing import List, Dict

ENVS: List[str] = [
    "highway",
    "roundabout",
    "merge",
    "exit",
    "racetrack",
    "parking",
    "lane-keeping",
]

K = "Kinematics"
GS = "GrayscaleObservation"
TTC = "TimeToCollision"
KG = "KinematicsGoal"
E = "ExitObservation"
A = "AttributesObservation"
OG = "OccupancyGrid"
MD_LIDAR = "lidar"
MD_IMAGE = "image"  # to use this observation, would require continuous hyperparameters for CnnPolicy

# Some environments require custom observation configurations
ALLOW_OBS: Dict[str, List[str]] = {
    "highway": [K, GS, TTC],
    "roundabout": [K, GS, TTC],
    "merge": [K, GS, TTC],
    "racetrack": [OG],
    "parking": [KG],
    "exit": [E],
    "lane-keeping": [A],
}

D, C = "Discrete", "Continuous"
ENV_ACTION_SPACE = {
    "highway": D,
    "roundabout": D,
    "merge": D,
    "exit": D,
    "racetrack": C,
    "parking": C,
    "lane-keeping": C,
}
