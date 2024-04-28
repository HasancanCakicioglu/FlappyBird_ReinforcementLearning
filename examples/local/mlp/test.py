import time

import numpy as np
from gymnasium.vector.utils import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, StackedObservations

from src.local.mlp.flappy_bird_local_env import FlappyBirdMlpLocalEnv

env = FlappyBirdMlpLocalEnv()


# Load the trained model
model = PPO.load("src/checkpoints/best_model_200000.zip", env=env)
#model = DQN.load("src/checkpoints/best_model_700000.zip", env=env)

for episode in range(100000):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        #action = env.action_space.sample()
        action, _states = model.predict(obs[0])
        observation, reward, done, _,info = env.step(action)
        total_reward += reward
        env.render("human")