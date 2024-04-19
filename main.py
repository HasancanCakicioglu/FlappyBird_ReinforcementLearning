from stable_baselines3 import DQN

from src.web.flappy_bird_env import FlappyBirdEnv

env = FlappyBirdEnv()

model = DQN.load("examples/web/src/checkpoints/best_model_28000.zip")

for episode in range(30):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        #action = env.action_space.sample()
        action, _states = model.predict(obs[0])
        observation, reward, done, _,info = env.step(action)
        total_reward += reward
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")