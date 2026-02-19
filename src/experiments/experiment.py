import gymnasium as gym
import highway_env
import highway_env.envs.intersection_env
import time
gym.register_envs(highway_env)
"""
Tested configs:

highway-v0
- vehicles_count: 100
- vehicles_density: 1, 2, (2.5, 3 - shouldn't be used)
- vehicles
"""
env = gym.make("highway-v0", render_mode="human") 
env.unwrapped.configure({
    "action": {
        "type": "DiscreteMetaAction",
        # "type": "DiscreteAction",
        # "type": "ContinuousAction",
    },
    "manual_control": True,    
    "lanes_count": 2,
    "vehicles_count": 35,
    "vehicles_density": 1.75,
    "duration": 30,
    "ego_spacing": 2.5,
})
env.reset()

done = False
while not done:
    env.step(env.action_space.sample())  # with manual control, these actions are ignored
