import os 
import gymnasium as gym
import panda_gym
import optuna
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import HerReplayBuffer
from sb3_contrib import TQC

def sample_hypeparams(trail):
    pass

def target_env(trail):
    try:
        env_id = "PandaPickAndPlace-v3"
        num_vec = 4
        train_time = 250_000
        env_opt = make_vec_env(env_id=env_id, n_envs=num_vec)
        env_norm = VecNormalize(env_opt)
        dir = os.path.curdir
        opt_log_dir = os.path.joint(dir, "logs", "optuna", f"{env_id}_{train_time}_{num_vec}")
        tensor_log_dir = os.path.joint(dir, "logs", "tensor", f"TQC_{env_id}_{train_time}_{num_vec}")
        model_params = sample_hypeparams(trail)
        train_model = TQC(
            **model_params,
            policy="MultiInputPolicy",
            env=env_norm,
            replay_buffer_class=HerReplayBuffer,
            tensorboard_log=tensor_log_dir,
            verbose=0
        )
        train_model.learn(total_timesteps=train_time)
        mean_reward, _ = evaluate_policy(
            train_model,
            env_norm,
            n_eval_episodes=50,
            render=True
        )
        return mean_reward
    except KeyboardInterrupt:
        print("UserInterupted")
        pass

def main():
    pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)