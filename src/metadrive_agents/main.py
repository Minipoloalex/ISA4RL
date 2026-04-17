
from gymnasium.envs.registration import register

def register_env():
    register(
        id="metadrive-v0",
        entry_point="metadrive.envs.my_metadrive_env.MyMetaDriveEnv",
    )

