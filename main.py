import time

from stable_baselines3 import PPO

from src.web.cnn.flappy_bird_env import FlappyBirdEnv
from src.web.mlp.flappy_bird_mlp_env import FlappyBirdMlpEnv

env = FlappyBirdMlpEnv()

for episode in range(100000):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        #action, _states = model.predict(obs[0])
        observation, reward, done, _,info = env.step(action)
        total_reward += reward
        env.render()


    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")