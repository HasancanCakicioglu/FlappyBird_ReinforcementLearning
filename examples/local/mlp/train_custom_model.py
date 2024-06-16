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
env = FlappyBirdMlpLocalEnv(env_config=2)

# Monitor wrapper ile env sarmalama
env = Monitor(env, LOG_DIR)

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[
        dict(pi=[64,128,64], vf=[64,128,64])  # Politika ve Değer ağları için büyük mimari
    ]
)

model = PPO("MlpPolicy", env, verbose=1 ,tensorboard_log=LOG_DIR)
#model = PPO.load("src/saved_models/best_model_ppo_1_000_000.zip",ent_coef=0.01,env=env,learning_rate=0.0005,policy_kwargs=policy_kwargs,verbose=1,tensorboard_log=LOG_DIR)
#model = PPO("MlpPolicy",ent_coef=0.01,env=env,learning_rate=0.0003,policy_kwargs=policy_kwargs,verbose=1,tensorboard_log=LOG_DIR)
#model = PPO.load("src/saved_models/custom_model_ppo_1_000_000.zip",ent_coef=0.01,env=env,learning_rate=0.0003,verbose=1,tensorboard_log=LOG_DIR)
print(model.policy)

model.learn(total_timesteps=2_000_000, callback=callback,tb_log_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
model.save("src/saved_models/custom_model_6obs_ppo_2_000_000.zip")