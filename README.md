# panda_rl_env
This Repo help with scripts to understand train with panda_gym library

## Libraries used

```
gymnasium
stable-baselines3
sb3-contrib
panda-gym
```

## Existing Envs in panda_gym
```
'PandaReach-v3', 'PandaReachJoints-v3', 'PandaReachDense-v3', 'PandaReachJointsDense-v3'

'PandaSlide-v3', 'PandaSlideJoints-v3', 'PandaSlideDense-v3', 'PandaSlideJointsDense-v3'

'PandaPush-v3', 'PandaPushJoints-v3', 'PandaPushDense-v3', 'PandaPushJointsDense-v3'

'PandaPickAndPlace-v3', 'PandaPickAndPlaceJoints-v3', 'PandaPickAndPlaceDense-v3', 'PandaPickAndPlaceJointsDense-v3'

'PandaStack-v3', 'PandaStackJoints-v3', 'PandaStackDense-v3', 'PandaStackJointsDense-v3'

'PandaFlip-v3', 'PandaFlipJoints-v3', 'PandaFlipDense-v3', 'PandaFlipJointsDense-v3'
```

## Create the virtual environment python
```sh
# For Windows
python -m venv .venv

# For Linux
python3 -m venv .venv

# Install all the requirements
pip install -r requirements.txt

```

## Run the training Reinforcement Learning Environment

```sh
python .\train\reach_tqc.py

```