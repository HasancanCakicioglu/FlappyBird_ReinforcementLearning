from stable_baselines3 import DQN
from src.config import CHECKPOINT_DIR, LOG_DIR
from src.callbacks.train_and_logging_callback import TrainAndLoggingCallback
from src.web.flappy_bird_env import FlappyBirdEnv

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR, verbose=1)
env = FlappyBirdEnv()


model = DQN("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, buffer_size=100_000, learning_starts=1000)

model.learn(total_timesteps=100_000, callback=callback)