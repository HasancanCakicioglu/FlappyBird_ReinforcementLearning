from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, StackedObservations

from src.local.mlp.flappy_bird_local_env import FlappyBirdMlpLocalEnv

env = FlappyBirdMlpLocalEnv()

#env = DummyVecEnv([lambda: env])

# Load the trained model
model = PPO.load("src/saved_models/best_model_ppo_1_000_000.zip", env=env,tensorboard_log="C:/Users/hckec/PycharmProjects/FlappyBird_ReinforcementLearning/examples/local/mlp/src/logs/2024-05-16_18-14-24_1")
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
        env.render("human")
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")