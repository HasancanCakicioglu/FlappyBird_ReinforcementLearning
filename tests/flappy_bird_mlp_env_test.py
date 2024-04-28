from stable_baselines3.common.env_checker import check_env

from src.web.mlp.flappy_bird_mlp_env import FlappyBirdMlpEnv

env = FlappyBirdMlpEnv()
check_env(env)