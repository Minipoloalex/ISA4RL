import os
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt

def get_grayscale_observation(scaling_value: float):
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": tuple([84, 84]),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": scaling_value,
        },
        "offscreen_rendering": True,
    }
    
    env.unwrapped.configure(config)
    env.reset()
    
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(1)
        if terminated or truncated:
            print("Bad")
            
    env.close()
    return obs

def compare_scalings(scaling_values: list[float]):
    num_plots = len(scaling_values)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # Ensure axes is iterable even if only one value is passed
    if num_plots == 1:
        axes = [axes]
        
    for i, val in enumerate(scaling_values):
        obs = get_grayscale_observation(val)
        print(f"Scaling {val} Min: {obs.min()}, Max: {obs.max()}")
        
        axes[i].imshow(obs[0], cmap='gray')
        axes[i].set_title(f"Scaling: {val}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

scaling_list = [1.0, 1.25, 1.5, 1.75, 2.0, None]
compare_scalings(scaling_list)
