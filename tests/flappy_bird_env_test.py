from stable_baselines3.common.env_checker import check_env

from src.web.cnn.flappy_bird_env import FlappyBirdEnv

env = FlappyBirdEnv()
check_env(env)