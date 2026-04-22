import gymnasium as gym
import highway_env
import copy
import traceback

def _env_viewer_deepcopy(self, memo):
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    
    new_env = copy.deepcopy(self.env, memo)
    new_config = copy.deepcopy(self.config, memo)
    new_config["offscreen_rendering"] = True
    
    result.__init__(new_env, config=new_config)
    return result

def main():
    # Make an environment with GrayscaleObservation to trigger the error
    env = gym.make("merge-v0", render_mode="rgb_array", config={
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140]
        }
    })
    env.reset()
    
    print("Trying default deepcopy...")
    try:
        copy.deepcopy(env)
        print("Success default deepcopy!")
    except Exception as e:
        print("Failed default deepcopy! (Expected)")
        traceback.print_exc()

    print("\nTrying with monkey-patched EnvViewer.__deepcopy__...")
    from highway_env.envs.common.graphics import EnvViewer
    original_deepcopy = getattr(EnvViewer, "__deepcopy__", None)
    EnvViewer.__deepcopy__ = _env_viewer_deepcopy
    try:
        new_env = copy.deepcopy(env)
        print("Success patched deepcopy!")
        # test if observation works
        new_env.step(0)
        print("Success stepping the copied env!")
    except Exception as e:
        print("Failed patched deepcopy!")
        traceback.print_exc()
    finally:
        if original_deepcopy is None:
            del EnvViewer.__deepcopy__
        else:
            EnvViewer.__deepcopy__ = original_deepcopy

if __name__ == "__main__":
    main()
