from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from src.config import CHECKPOINT_DIR, LOG_DIR
from src.callbacks.train_and_logging_callback import TrainAndLoggingCallback
from src.web.mlp.flappy_bird_mlp_env import FlappyBirdMlpEnv

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR, verbose=1)
env = FlappyBirdMlpEnv()

# Monitor wrapper ile env sarmalama
env = Monitor(env, LOG_DIR)

# Vec env wrapper ile env sarmalama
env = DummyVecEnv([lambda: env])

model = DQN("MlpPolicy", env, verbose=1,tensorboard_log=LOG_DIR,learning_rate=0.0003)

model.learn(total_timesteps=100_000, callback=callback)
