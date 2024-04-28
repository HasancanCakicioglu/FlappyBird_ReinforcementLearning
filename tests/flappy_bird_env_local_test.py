from stable_baselines3.common.env_checker import check_env
from src.local.flappy_bird_env_local import FlappyBirdEnvLocal


env = FlappyBirdEnvLocal()
check_env(env)