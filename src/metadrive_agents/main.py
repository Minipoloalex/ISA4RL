from gymnasium.envs.registration import register
from panda3d.core import loadPrcFileData

import metadrive
from methods.main import main
from common.env_utils import METADRIVE_ENVS

loadPrcFileData("", "notify-level-linmath error")

if __name__ == "__main__":
    main(METADRIVE_ENVS)
