from datetime import datetime
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from src.config import CHECKPOINT_DIR, LOG_DIR
from src.callbacks.train_and_logging_callback import TrainAndLoggingCallback
from src.local.mlp.flappy_bird_local_env import FlappyBirdMlpLocalEnv

# Optuna tarafından bulunan en iyi hiperparametreler
best_hyperparameters = {
    'learning_rate': 1.7484385994222768e-05,
    'n_steps': 368,
    'gamma': 0.9931833599871185,
    'gae_lambda': 0.8761865939326113,
    'ent_coef': 4.9572375032109966e-08,
    'clip_range': 0.3879429173159942,
    'vf_coef': 0.7365801485050952,
    'max_grad_norm': 1.1314962287108825,
    'net_arch': 'medium'  # 'medium' olarak belirtilmiş ama aşağıda net_arch ayarlamasına dikkat etmeliyiz
}

# Ağ mimarisi seçenekleri, 'medium' için tanımlanan yapı
net_arch_options = {
    'small': [64, 64],
    'medium': [128, 128],
    'large': [256, 256, 256]
}

# Modelin politikası için kullanılan ek argümanlar
policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[
        dict(pi=net_arch_options[best_hyperparameters['net_arch']], vf=net_arch_options[best_hyperparameters['net_arch']])
    ]
)

# Çalıştırma ortamını kurma
env = FlappyBirdMlpLocalEnv()

# Monitor wrapper ile env sarmalama
env = Monitor(env, LOG_DIR)

# Eğitim ve kayıt callback'i
callback = TrainAndLoggingCallback(check_freq=100_000, save_path=CHECKPOINT_DIR, verbose=1)

# PPO modelini en iyi hiperparametrelerle oluşturma
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=best_hyperparameters['learning_rate'],
    n_steps=best_hyperparameters['n_steps'],
    gamma=best_hyperparameters['gamma'],
    gae_lambda=best_hyperparameters['gae_lambda'],
    ent_coef=best_hyperparameters['ent_coef'],
    clip_range=best_hyperparameters['clip_range'],
    vf_coef=best_hyperparameters['vf_coef'],
    max_grad_norm=best_hyperparameters['max_grad_norm'],
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=LOG_DIR
)

# Modelin eğitimini başlatma
model.learn(total_timesteps=2_000_000, callback=callback, tb_log_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print(model.policy)

# Modeli kaydetme
model.save("src/saved_models/custom_model_best_params_ppo_2_000_000.zip")
