import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3 import HerReplayBuffer, SAC, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import argparse
import highway_env  # noqa: F401
import time

# ==================================
#        Main script
# ==================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the environment to experiment with training."
    )
    # Options: highway-v0, merge-v0, roundabout-v0, exit-v0, lane-keeping-v0, parking-v0
    #           two-way-v0, u-turn-v0, racetrack-v0, racetrack-large-v0, racetrack-oval-v0
    parser.add_argument("--env", default="highway-v0", help="Environment ID to load.")
    parser.add_argument("--train", dest="train", action="store_true", help="Disable training")
    parser.set_defaults(train=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    env_id = args.env
    train = args.train
    config = {}
    config["render_mode"] = None
    N_TRAIN_ENVS = 1
    EVAL_FREQ = int(4e3)
    policy_arch = "MlpPolicy"
    train_timesteps = int(1e5)
    callback_eval_episodes = 10
    if env_id == "u-turn-v0":
        config["duration"] = 25
    elif env_id == "exit-v0":
        """
        Rewards: 18 steps
        Default values: goal_reward = 1, high_speed_reward: 0.1, right_lane_reward: 0, collision_reward: 0
        """
        config["lanes_count"] = 4
        config["duration"] = 20
        config["high_speed_reward"] = 0.01
    elif env_id == "lane-keeping-v0":
        policy_arch = "MultiInputPolicy"
    elif env_id.startswith("racetrack"):
        config["duration"] = 60
        train_timesteps = int(2e5)
        # config["collision_reward"] = -10
        # config["action_reward"] = -0.1
    elif env_id == "parking-v0":
        policy_arch = "MultiInputPolicy"
        train_timesteps = int(2e5)
        config["duration"] = 60
        config["add_walls"] = True
        config["parking_spots"] = 4
        config["vehicles_count"] = 4
    env_kwargs = {"config": config}
    if train:
        eval_env = make_vec_env(env_id, vec_env_cls=DummyVecEnv, env_kwargs={"config": config})
        if env_id == "parking-v0":
            eval_env = VecNormalize(eval_env, training=False, norm_reward=False, norm_obs=True)

        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=callback_eval_episodes,
            best_model_save_path=f"result_experiment_{env_id}/best_model",
            log_path=f"result_experiment_{env_id}/",
            eval_freq=EVAL_FREQ // N_TRAIN_ENVS,
            deterministic=True,
            render=False,
        )
        n_cpu = 6
        batch_size = 64
        # env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        # env = make_vec_env("highway-fast-v0", n_envs=16, vec_env_cls=SubprocVecEnv)
        train_env = make_vec_env(
            env_id,
            n_envs=N_TRAIN_ENVS,
            env_kwargs=env_kwargs,
            vec_env_cls=DummyVecEnv,
            monitor_dir=f"result_experiment_{env_id}/",
        )
        if env_id == "parking-v0":
            train_env = VecNormalize(train_env, training=True, norm_obs=True, norm_reward=True)
        # config = {
        #     "n_envs": 16,
        #     "n_timesteps": 100000,
        #     "policy": "MlpPolicy",
        #     "n_steps": 16,
        #     "gae_lambda": 0.98,
        #     "gamma": 0.99,
        #     "n_epochs": 4,
        #     "ent_coef": 0.0,
        # }
        # SAC hyperparams:
        # model = SAC(
        #     "MultiInputPolicy",
        #     train_env,
        #     replay_buffer_class=HerReplayBuffer,
        #     replay_buffer_kwargs=dict(
        #         n_sampled_goal=4,
        #         goal_selection_strategy="future",
        #     ),
        #     verbose=1,
        #     buffer_size=int(1e6),
        #     learning_rate=1e-3,
        #     gamma=0.95,
        #     batch_size=256,
        #     policy_kwargs=dict(net_arch=[256, 256, 256]),
        # )
        model = PPO(
            policy_arch,
            train_env,
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
            verbose=1,
            tensorboard_log=f"result_experiment_{env_id}/",
            device="cpu",
        )
        # Train the agent
        model.learn(total_timesteps=train_timesteps, progress_bar=True, callback=eval_callback)
        # Save the agent
        model.save(f"result_experiment_{env_id}/model")

        train_env.close()
        eval_env.close()

    print(">> Evaluating: showing visualization")
    model = PPO.load(f"result_experiment_{env_id}/best_model/best_model", device="cpu")
    env = gym.make(env_id, config=config, render_mode="human")
    env = Monitor(env)
    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            time.sleep(0.1)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
