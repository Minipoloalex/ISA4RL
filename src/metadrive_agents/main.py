
from gymnasium.envs.registration import register
from methods.main import main
from common.env_utils import METADRIVE_ENVS

def register_env():
    register(
        id="metadrive-v0",
        entry_point="metadrive.envs.my_metadrive_env.MyMetaDriveEnv",
    )

if __name__ == "__main__":
    register_env()
    main(METADRIVE_ENVS)
