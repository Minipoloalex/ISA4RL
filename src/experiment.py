import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import highway_env  # noqa: F401


# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 6
        batch_size = 64
        # env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        # env = make_vec_env("highway-fast-v0", n_envs=16, vec_env_cls=SubprocVecEnv)
        env = make_vec_env("highway-fast-v0", n_envs=1, vec_env_cls=DummyVecEnv)
        # config = {
        #     "n_envs": 16,
        #     "n_timesteps": 1000000.0,
        #     "policy": "MlpPolicy",
        #     "n_steps": 16,
        #     "gae_lambda": 0.98,
        #     "gamma": 0.99,
        #     "n_epochs": 4,
        #     "ent_coef": 0.0,
        # }
        model = PPO(
            "MlpPolicy",
            env,
            # policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            # n_steps=batch_size * 12 // n_cpu,
            # batch_size=batch_size,
            # n_epochs=10,
            # learning_rate=5e-4,
            # gamma=0.8,
            # verbose=2,
            # n_steps=16,
            # gae_lambda=0.98,
            # gamma=0.99,
            # n_epochs=4,
            # ent_coef=0.0,
            tensorboard_log="highway_ppo4/",
        )
        # Train the agent
        model.learn(total_timesteps=int(5e4), progress_bar=True)
        # Save the agent
        model.save("highway_ppo4/model")

    model = PPO.load("highway_ppo4/model")
    env = gym.make("highway-fast-v0", render_mode="human")
    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
