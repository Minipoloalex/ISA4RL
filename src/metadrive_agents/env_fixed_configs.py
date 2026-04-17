from copy import deepcopy

from metadrive.envs.metadrive_env import METADRIVE_DEFAULT_CONFIG as _METADRIVE_DEFAULT_CONFIG

METADRIVE_FIXED_CONFIGS = {
    "env_id": "metadrive-v0",
    "train_timesteps": int(1e6),
    "eval_freq": int(1e4),
    "n_eval_episodes": 5,
    "n_test_episodes": 10,
    "config": { # when instantiating the config, need to be careful with recursive dicts
        "random_traffic": False,
        "num_scenarios": int(1e3),
        "map": None,
    },
}
