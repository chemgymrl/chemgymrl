import sys
import os
import csv
sys.path.append('../')
sys.path.append('../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np
from gym import envs
from stable_baselines import A2C
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy


all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if 'React' in env_spec.id]
print(env_ids)

with open('A2Creactionbench.csv', 'w+') as myfile:
    myfile.write('{0},{1}\n'.format("Episode", "A2C"))

log_dir = "./tmp/"
env = gym.make('WurtzReact-v1')
env = Monitor(env, log_dir, allow_early_resets=True)
action_set = ['Temperature', 'Volume', "1-chlorohexane", "2-chlorohexane", "3-chlorohexane", "Na"]

assert len(action_set) == env.action_space.shape[0]

print("The action space is", env.action_space)

model = A2C("MlpPolicy", env, verbose=1, seed=100)

os.makedirs(log_dir, exist_ok=True)
total_episodes_list = []
average_reward_list = [] 

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')




callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
time_steps = 25000
model.learn(total_timesteps=time_steps, callback=callback, log_interval=100)
#results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "A2C Reaction Bench")
x, y = ts2xy(load_results(log_dir), 'timesteps')
#print("The value of y are", y)
fig = plt.figure("A2C REaction Bench")
plt.plot(x,y)
plt.xlabel('Number of Timesteps')
plt.ylabel('Rewards')
#results_plotter.plot_curves([x,y], results_plotter.X_TIMESTEPS, "A2C REaction bench")
#print("the value of y are", y)
title = "A2C Reaction Bench"
plt.title(title)
plt.savefig('./A2Creactionbench.png')
