import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.callbacks.train_and_logging_callback import TrainAndLoggingCallback
from src.config import CHECKPOINT_DIR, LOG_DIR
from src.local.mlp.flappy_bird_local_env import FlappyBirdMlpLocalEnv
from src.wrappers.env_stack import StackedObservations
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn

callback = TrainAndLoggingCallback(check_freq=100_000, save_path=CHECKPOINT_DIR, verbose=1)
env = FlappyBirdMlpLocalEnv()


N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3


DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": env,
}


def sample_ppo_params(trial: optuna.Trial) -> dict:
    """Sampler for PPO hyperparameters."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 256, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "ent_coef": ent_coef,
        "gamma": gamma
    }


def objective(trial: optuna.Trial) -> float:
    kwargs = sample_ppo_params(trial)
    model = PPO("MlpPolicy", env, **kwargs)
    eval_env = Monitor(env)
    eval_callback = EvalCallback(eval_env, best_model_save_path=None, log_path=None, eval_freq=EVAL_FREQ,
                                 deterministic=True, render=False)

    try:
        model.learn(total_timesteps=N_TIMESTEPS, callback=eval_callback)
        mean_reward = np.mean(eval_callback.best_mean_reward)
    except AssertionError:
        mean_reward = float("-inf")
    finally:
        model.env.close()
        eval_env.close()

    return mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
