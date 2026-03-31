from typing import List, Dict

ENVS: List[str] = [
    "highway",
    "roundabout",
    "merge",
    "racetrack",
    "parking",
    "exit",
    "lane-keeping",
]

K = "Kinematics"
GS = "GrayscaleObservation"
TTC = "TimeToCollision"
KG = "KinematicsGoal"
E = "ExitObservation"
A = "AttributesObservation"
OG = "OccupancyGrid"

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
