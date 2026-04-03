__DEFAULT_TRAIN_CONFIGS = {
    "train_timesteps": int(1e5),
    "eval_freq": int(1e3),
    "n_eval_episodes": 10,
    "n_test_episodes": 10,
}

HIGHWAY_FIXED_CONFIGS = {
    "env_id": "highway-fast-v0",
    "config": {},
    **__DEFAULT_TRAIN_CONFIGS,
}
MERGE_FIXED_CONFIGS = {
    "env_id": "merge-generic-v0",
    "config": {
        "length_after_merge": 250,
    },
    **__DEFAULT_TRAIN_CONFIGS,
}
ROUNDABOUT_FIXED_CONFIGS = {
    "env_id": "roundabout-generic-v0",
    "config": {},
    **__DEFAULT_TRAIN_CONFIGS,
}
U_TURN_FIXED_CONFIGS = {
    "env_id": "u-turn-v0",
    "config": {"duration": 25},
    **__DEFAULT_TRAIN_CONFIGS,
}
TWO_WAY_FIXED_CONFIGS = {
    "env_id": "two-way-v0",
    "config": {},
    **__DEFAULT_TRAIN_CONFIGS,
}
EXIT_FIXED_CONFIGS = {
    "env_id": "exit-v0",
    "config": {"high_speed_reward": 0.01},
    **__DEFAULT_TRAIN_CONFIGS,
}
LANE_KEEPING_FIXED_CONFIGS = {
    "env_id": "lane-keeping-v0",
    "config": {
        "action": { # steering_range set later
            "type": "ContinuousAction",
            "longitudinal": False,
            "lateral": True,
            "dynamical": True,
        },
    },
    "train_timesteps": int(2e5),
    "eval_freq": int(2e3),
    "n_eval_episodes": 10,
    "n_test_episodes": 10,
}
BASIC_RACETRACK_FIXED_CONFIGS = {
    "env_id": "racetrack-v0",
    "config": {
        "duration": 100,
        "action": { # no steering_range here
            "type": "ContinuousAction",
            "longitudinal": False,
            "lateral": True,
        },
    },
    "train_timesteps": int(2e5),
    "eval_freq": int(2e3),
    "n_eval_episodes": 5,
    "n_test_episodes": 10,
}
OVAL_RACETRACK_FIXED_CONFIGS = {
    "env_id": "racetrack-oval-v0",
    "config": {
        "duration": 100,
        "action": { # no steering_range here
            "type": "ContinuousAction",
            "longitudinal": False,
            "lateral": True,
        },
    },
    "train_timesteps": int(2e5),
    "eval_freq": int(2e3),
    "n_eval_episodes": 5,
    "n_test_episodes": 10,
}
PARKING_FIXED_CONFIGS = {
    "env-id": "parking-v0",
    "train_timesteps": int(3e5),
    "eval_freq": int(3e3),
    "n_eval_episodes": 5,
    "n_test_episodes": 10,
    "config": {
        "duration": 60,
        "action": { # steering_range not here
            "type": "ContinuousAction",
        },
    },
}
