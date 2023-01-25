"""
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.

Module to train an RL agent on a gym. Options are handled with the Opt() class.

Usage from a command line works as follows:
>>python RLTrain.py <option1 name>=<option1 value> <option2 name>=<option2 value>. . .

Example call:
>>python RLTrain.py steps=20000 algorithm=A2C

:title: RLTrain.py

:author: Kyle Sprague

:history: 22-01-2023
"""


class Opt:
    """
    Description of these options:
    
    policy  (str) -- which policy to use in your network (MLP, etc)
    algorithm  (str) -- Number of minibatches in the queue (PPO,DQN,etc)
    environment (str) -- Which environment to use
    steps (int) -- Number of training steps
    dir (str) -- Output directory, set to <DEFAULT> for a generated folder name
    seed (int) -- random seed for the run
    """
    DEFAULTS={"policy":"MlpPolicy","algorithm":"PPO2","environment":"WurtzReact-v1","steps": 500,"dir":"<DEFAULT>","seed":None}
    def __init__(self,**kwargs):
        self.__dict__.update(Opt.DEFAULTS)
        self.__dict__.update(kwargs)

    def __str__(self):
        out=""
        for key in self.__dict__:
            line=key+" "*(30-len(key))+ "\t"*3+str(self.__dict__[key])
            out+=line+"\n"
        return out
    def cmd(self):
        """Returns a string with command line arguments corresponding to the options"""
        out=""
        for key in self.__dict__:
            line=key+"="+str(self.__dict__[key])
            out+=line+" "
        return out[:-1]
    
    def apply(self,args):
        """Takes in a set of command line arguments and turns them into options"""
        kwargs = dict()
        for arg in args:
            try:
                key,val=arg.split("=")
                kwargs[key]=self.sus_cast(val)
            except:pass
        self.__dict__.update(kwargs)
    def sus_cast(self,x0):
        """Casting from a string to a floating point or integer
        TODO: add support for booleans
        """
        try:
            if x0=="True":return True
            elif x0=="False":return False
            elif x0=="None":return None
            x=x0
            x=float(x0)
            x=int(x0)
        except:return x
        return x
    def from_file(self,fn):
        """Takes a file and converts it to options"""
        kwargs = dict()
        with open(fn,"r") as f:
          for line in f:
            line=line.strip()
            split = line.split("\t")
            key,val = split[0].strip(),split[-1].strip()
            try:
                kwargs[key]=self.sus_cast(val)
            except:pass
        self.__dict__.update(kwargs)




from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
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
        pass
        #if self.save_path is not None:
            #os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                #clear_output(wait=True)
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





import stable_baselines
from stable_baselines import A2C
from stable_baselines import PPO2
import gym
import chemistrylab
from stable_baselines.bench import Monitor
import sys
import os
import numpy as np

ALGO={"PPO2":PPO2,"A2C":A2C}


if __name__=="__main__":


    #get options
    print(sys.argv[1:])
    op=Opt()
    op.apply(sys.argv[1:])

    print(op)
    
    if op.dir=="<DEFAULT>":
        op.dir=op.algorithm+"_"+op.environment
    
    
    #Set up output directory
    os.makedirs(op.dir, exist_ok=True)
    
    
    #save the settings
    with open(op.dir+"/settings.txt","w") as f:
        f.write(str(op)+"\n")
        
    #set up logging of print statements
    old_stdout = sys.stdout
    log_file = open(op.dir+"/output.log","w")
    sys.stdout = log_file
    
    #make the environment
    env = gym.make(op.environment)
    print("The action space is", env.action_space)
    print("The observation space is", env.observation_space)
    
    #set up environment monitor
    env = Monitor(env, op.dir, allow_early_resets=True)

    #initialize model
    model = ALGO[op.algorithm](op.policy, env, verbose=1, seed = op.seed)
    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=op.dir)
    #train the model
    model.learn(total_timesteps=op.steps, callback=callback, log_interval=100)

    
    #Clean up the logging
    sys.stdout = old_stdout
    log_file.close()