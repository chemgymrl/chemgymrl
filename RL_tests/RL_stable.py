from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import gym
from wandb.integration.sb3 import WandbCallback
import wandb
import chemistrylab

config = {
    "rl_alg": PPO,
    "policy_type": "MlpPolicy",
    "policy_name": "PPO_React",
    "run_name": "FictReact-PPO",
    "total_timesteps": 5000,
    "log_int": 1,
    "env_name": "FictReact-v1",
    "policy_kwargs": dict(net_arch=[32, 32])
}

run = wandb.init(
    project="ChemGymRL",
    entity="clean",
    config=config,
    sync_tensorboard=True,
    save_code=True,
)
wandb.run.name = config["run_name"]
wandb.run.save()

def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])
model = config["rl_alg"](config["policy_type"], env, n_steps=128, n_epochs=5, policy_kwargs=config["policy_kwargs"], verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    log_interval=config["log_int"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
model.save(config["policy_name"])
run.finish()
