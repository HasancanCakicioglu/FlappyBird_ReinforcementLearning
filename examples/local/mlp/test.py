import time
from random import random

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, StackedObservations

from src.local.mlp.flappy_bird_local_env import FlappyBirdMlpLocalEnv

env = FlappyBirdMlpLocalEnv(env_config=2)

#env = DummyVecEnv([lambda: env])

# Load the trained model
model = PPO.load("src/saved_models/custom_model_6obs_ppo_2_000_000.zip", env=env)
#model = DQN.load("src/saved_models/dqn_5_000_000.zip", env=env)


for episode in range(100000):
    observation = env.reset()[0]
    done = False
    total_reward = 0


    while not done:
        #action = env.action_space.sample()
        action, _states = model.predict(observation,deterministic=True)
        observation, reward, done, _,info = env.step(action)
        total_reward += reward
        env.render("human")
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")