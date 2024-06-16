from datetime import datetime

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
from src.config import CHECKPOINT_DIR, LOG_DIR
from src.callbacks.train_and_logging_callback import TrainAndLoggingCallback
from src.local.mlp.flappy_bird_local_env import FlappyBirdMlpLocalEnv
from src.wrappers.env_stack import StackedObservations
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
import torch as th

callback = TrainAndLoggingCallback(check_freq=100_000, save_path=CHECKPOINT_DIR, verbose=1)
env = FlappyBirdMlpLocalEnv()

# Monitor wrapper ile env sarmalama
env = Monitor(env, LOG_DIR)


# Wrap the environment with StackedObservations
# Vec env wrapper ile env sarmalama
env = DummyVecEnv([lambda: env])


# Frame stacking parameters
num_envs = 1  # Since you're using a single environment
n_stack = 20  # Number of frames to stack
env = VecFrameStack(env, n_stack)

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[
        dict(pi=[256, 256, 256], vf=[256, 256, 256])  # Politika ve Değer ağları için büyük mimari
    ]
)

model = PPO("MlpPolicy", env, verbose=1,policy_kwargs=policy_kwargs ,tensorboard_log=LOG_DIR,learning_rate=0.0003)
print(model.policy)

model.learn(total_timesteps=1_000_000, callback=callback,tb_log_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
model.save("src/saved_models/custom_model_ppo_stacked20_1_000_000.zip")