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