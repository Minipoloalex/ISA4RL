from metadrive import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.map.pg_map import MapGenerateMethod
import matplotlib.pyplot as plt
from metadrive import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
import logging

MAPS = ["SC", "rORY", "TXT"]

for mp in MAPS:
    print(f"Map: {mp}")
    env = MetaDriveEnv(dict(num_scenarios=1, map=mp, log_level=logging.WARNING))
    plt.tight_layout(pad=-3)

    env.reset(seed=0)
    m = draw_top_down_map(env.current_map)
    plt.imshow(m, cmap="bone")
    plt.xticks([])
    plt.yticks([])
    env.close()
    plt.savefig(f"image_{mp}.pdf", dpi=1000, bbox_inches="tight")
    plt.close()
