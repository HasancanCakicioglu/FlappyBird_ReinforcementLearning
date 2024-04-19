import time

from stable_baselines3 import DQN, PPO

from src.web.flappy_bird_env import FlappyBirdEnv

env = FlappyBirdEnv()

#model = DQN.load("examples/web/src/checkpoints/best_model_28000.zip")
model = PPO.load("examples/web/src/checkpoints/best_model_100000.zip")

for episode in range(100000):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        #action = env.action_space.sample()
        action, _states = model.predict(obs[0])
        observation, reward, done, _,info = env.step(action)
        total_reward += reward
        env.render()

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")