import os 
import gymnasium as gym
import panda_gym
import optuna
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3 import HerReplayBuffer
from sb3_contrib import TQC

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable to PyTorch. Install a CUDA-enabled PyTorch build "
        "and verify torch.cuda.is_available() before starting training."
    )

print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

def sample_hypeparams(trail):
    params = {}
    policy_params = {}
    # params["n_steps"] = trail.suggest_int("n_steps", 2048, 14336)
    params["buffer_size"] = trail.suggest_categorical("buffer_size", [50_000, 100_000, 250_000, 500_000])
    params["learning_rate"] = trail.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    params["batch_size"] = trail.suggest_categorical("batch_size", [128, 258, 512])
    params["gamma"] = trail.suggest_uniform("gamma", 0.99, 0.999)
    params["tau"] = trail.suggest_uniform("tau", 0.001, 0.05)
    policy_params["net_arch"] = trail.suggest_categorical("net_arch", [
        [256, 256],        
        [256, 256, 256],
        [512, 512],
        [512, 512, 512],
    ])
    policy_params["n_critics"] = trail.suggest_int("n_critics", 2, 4)
    policy_params["share_features_extractor"] = trail.suggest_categorical("share_features_extractor", [True, False])
    params["policy_kwargs"] = policy_params
    params["learning_starts"] = trail.suggest_float("learning_starts", 1e2, 1e3, log=True)
    print("=========================================")
    print(f"\n\n\n\tTrail Parameters {params}\n\n\n")
    print("=========================================")
    return params

def target_env(trail):
    try:
        env_id = "PandaReachDense-v3"
        num_vec = 5
        train_time = 250_000
        env_opt = make_vec_env(env_id=env_id, n_envs=num_vec)
        env_norm = VecNormalize(env_opt)
        max_episode_steps = getattr(env_opt.envs[0], "_max_episode_steps", None)
        if max_episode_steps is None:
            max_episode_steps = getattr(
                getattr(env_opt.envs[0], "spec", None), "max_episode_steps", 0
            )
        min_learning_starts = (max_episode_steps * num_vec) + 1
        dir = os.path.curdir
        opt_log_dir = os.path.join(dir, "logs", "optuna", f"{env_id}_{train_time}_{num_vec}")
        tensor_log_dir = os.path.join(dir, "logs", "tensor", f"TQC_{env_id}_{train_time}_{num_vec}")
        model_params = sample_hypeparams(trail)
        retry_count = 0
        while True:
            train_model = TQC(
                **model_params,
                policy="MultiInputPolicy",
                env=env_opt,
                replay_buffer_class=HerReplayBuffer,
                tensorboard_log=tensor_log_dir,
                verbose=0,
                device="cuda:0"
            )
            try:
                train_model.learn(total_timesteps=train_time)
                break
            except RuntimeError as error:
                if "nan" in str(error).lower() or "invalid values" in str(error).lower():
                    raise optuna.TrialPruned("Training diverged with non-finite policy values") from error
                if "Unable to sample before the end of the first episode" not in str(error):
                    raise
                retry_count += 1
                if retry_count > 3:
                    raise
                model_params["learning_starts"] = trail.suggest_int(
                    f"learning_starts_retry_{retry_count}",
                    max(100, min_learning_starts),
                    1_000
                )
            except ValueError as error:
                if "nan" in str(error).lower() or "invalid values" in str(error).lower():
                    raise optuna.TrialPruned("Training diverged with non-finite policy values") from error
                raise
        mean_reward, _ = evaluate_policy(
            train_model,
            env_norm,
            n_eval_episodes=20,
            render=False
        )
        return mean_reward
    except KeyboardInterrupt:
        print("UserInterupted")
        raise

def main():
    param_tunning = optuna.create_study(direction = "maximize")
    param_tunning.optimize(target_env, n_trials=5, n_jobs=1, show_progress_bar=True, gc_after_trial=True)
    print(f"Best HyperParams : \n{param_tunning.best_params} \nat this\n{param_tunning.best_trial} trial.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(0)
    except Exception as e:
        print(e)
