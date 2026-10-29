import json
import os
from datetime import datetime

import numpy as np
import optuna
import panda_gym
from stable_baselines3 import HerReplayBuffer, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize


ENV_ID = "PandaReachDense-v3"
NUM_ENVS = 5
TOTAL_TIMESTEPS = 150_000
N_TRIALS = 5
NETWORK_ARCHITECTURES = {
    "small": [128, 128],
    "medium": [256, 256],
    "large": [256, 256, 256],
}


def sample_hyperparameters(trial):
    architecture = trial.suggest_categorical(
        "net_arch", list(NETWORK_ARCHITECTURES)
    )
    return {
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [50_000, 100_000, 250_000, 500_000]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "gamma": trial.suggest_float("gamma", 0.99, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.05),
        "learning_starts": trial.suggest_int(
            "learning_starts", 100, 1_000, log=True
        ),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 4]),
        "ent_coef": trial.suggest_categorical("ent_coef", ["auto", "auto_0.1"]),
        "policy_kwargs": {"net_arch": NETWORK_ARCHITECTURES[architecture]},
    }


def target_env(trial):
    env = VecNormalize(
        make_vec_env(ENV_ID, n_envs=NUM_ENVS),
        norm_obs=True,
        norm_reward=False,
    )
    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            **sample_hyperparameters(trial),
            replay_buffer_class=HerReplayBuffer,
            tensorboard_log=os.path.join("logs", "tensor", f"SAC_{ENV_ID}"),
            verbose=0,
            device="auto",
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        env.training = False
        mean_reward, _ = evaluate_policy(
            model, env, n_eval_episodes=20, render=False
        )
        return mean_reward
    finally:
        env.close()


def save_results(study):
    best_trial = study.best_trial
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs",
        "best_trials",
    )
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(
        output_dir, f"SAC_{ENV_ID}_{timestamp}.npz"
    )
    trials = study.trials
    np.savez_compressed(
        output_path,
        trial_numbers=np.array([trial.number for trial in trials]),
        trial_values=np.array(
            [np.nan if trial.value is None else trial.value for trial in trials]
        ),
        trial_states=np.array([trial.state.name for trial in trials]),
        trial_params=np.array([json.dumps(trial.params) for trial in trials]),
        best_params=json.dumps(study.best_params),
        best_trial_number=best_trial.number,
        best_trial_value=best_trial.value,
    )
    print(
        f"Best trial: {best_trial.number} "
        f"(value={best_trial.value}, params={study.best_params})\n"
        f"Saved trial results to {output_path}"
    )


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        target_env,
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    save_results(study)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(0)
