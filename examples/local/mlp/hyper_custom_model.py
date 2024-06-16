import optuna
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Özel ortamınız için FlappyBirdMlpLocalEnv kullanabilirsiniz
from src.local.mlp.flappy_bird_local_env import FlappyBirdMlpLocalEnv


# Optuna objective fonksiyonunu tanımlıyoruz
def objective(trial):
    # Hyperparametreleri Optuna'dan öneriyoruz
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_int('n_steps', 64, 2048, log=True)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 1.0)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.4)
    vf_coef = trial.suggest_uniform('vf_coef', 0.1, 0.9)
    max_grad_norm = trial.suggest_uniform('max_grad_norm', 0.3, 5.0)
    net_arch = trial.suggest_categorical('net_arch', ['small', 'medium', 'large'])

    # Ağ mimarisi seçenekleri
    net_arch_options = {
        'small': [64, 64],
        'medium': [128, 128],
        'large': [256, 256, 256]
    }

    # Policy kwargs ayarları
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=net_arch_options[net_arch]
    )

    # Özel ortam
    env = FlappyBirdMlpLocalEnv()
    env = Monitor(env)  # Performans izleme için ortamı sarmalıyoruz

    # PPO modelini Optuna'dan gelen hyperparametrelerle oluşturuyoruz
    model = PPO('MlpPolicy', env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                clip_range=clip_range,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=0)

    # Modeli belirli bir adımda değerlendiriyoruz
    eval_env = FlappyBirdMlpLocalEnv()
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=10000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=100000, callback=eval_callback)

    # Modelin ortalama ödülünü değerlendiriyoruz
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)

    return mean_reward


# Optuna study'si oluşturma ve optimize etme
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# En iyi hyperparametreleri yazdırıyoruz
print('Best hyperparameters: ', study.best_params)
