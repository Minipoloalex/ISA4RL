DEFAULT_TRAIN_CONFIGS = {
    "train_timesteps": int(1e5),
    "eval_freq": int(1e3),
    "n_eval_episodes": 10,
}

HIGHWAY_FIXED_CONFIGS = {
    "env_id": "highway-fast-v0",
    "config": {},
    **DEFAULT_TRAIN_CONFIGS,
}
MERGE_FIXED_CONFIGS = {
    "env_id": "merge-v0",
    "config": {},
    **DEFAULT_TRAIN_CONFIGS,
}
ROUNDABOUT_FIXED_CONFIGS = {
    "env_id": "roundabout-v0",
    "config": {},
    **DEFAULT_TRAIN_CONFIGS,
}
U_TURN_FIXED_CONFIGS = {
    "env_id": "u-turn-v0",
    "config": {"duration": 25},
    **DEFAULT_TRAIN_CONFIGS,
}
TWO_WAY_FIXED_CONFIGS = {
    "env_id": "two-way-v0",
    "config": {},
    **DEFAULT_TRAIN_CONFIGS,
}
EXIT_FIXED_CONFIGS = {
    "env_id": "exit-v0",
    "config": {"high_speed_reward": 0.01},
    **DEFAULT_TRAIN_CONFIGS,
}
LANE_KEEPING_FIXED_CONFIGS = {
    "env_id": "lane-keeping-v0",
    "config": {},
    **DEFAULT_TRAIN_CONFIGS,
}
RACETRACK_FIXED_CONFIGS = {
    "env_id": "racetrack-v0",
    "config": {"duration": 60},
    "train_timesteps": int(2e5),
    "eval_freq": int(2e3),
    "n_eval_episodes": 5,
}
PARKING_FIXED_CONFIGS = {
    "env-id": "parking-v0",
    "config": {},
    # TODO: make parking work
}
