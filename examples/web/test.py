from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, StackedObservations

from src.web.cnn.flappy_bird_env import FlappyBirdEnv

env = FlappyBirdEnv()

#env = DummyVecEnv([lambda: env])

# Load the trained model
model = PPO.load("src/checkpoints/best_model_100000.zip", env=env)
#model = DQN.load("src/saved_models/best_model_dqn_1_000_000.zip", env=env)

for episode in range(100000):
    observation = env.reset()[0]
    done = False
    total_reward = 0


    while not done:
        #action = env.action_space.sample()
        action, _states = model.predict(observation)
        observation, reward, done, _,info = env.step(action)
        total_reward += reward
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")