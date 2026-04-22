import gymnasium as gym
import highway_env
import copy
from pprint import pprint

def find_surfaces(obj, path="", memo=None):
    if memo is None:
        memo = set()
    if id(obj) in memo:
        return
    memo.add(id(obj))
    
    if type(obj).__name__ == 'WorldSurface' or type(obj).__name__ == 'Surface':
        print(f"Found {type(obj).__name__} at {path}")
        return

    if type(obj).__name__ == 'EnvViewer':
        print(f"Found EnvViewer at {path}")
        # we still want to traverse inside the viewer just in case
        # but actually we just want to know who holds the viewer

    if isinstance(obj, dict):
        for k, v in obj.items():
            find_surfaces(v, f"{path}[{repr(k)}]", memo)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_surfaces(v, f"{path}[{i}]", memo)
    elif hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            find_surfaces(v, f"{path}.{k}", memo)

def main():
    env = gym.make("merge-v0", render_mode="rgb_array")
    env.reset()
    env.render() # trigger viewer creation
    
    find_surfaces(env, "env")

if __name__ == "__main__":
    main()
