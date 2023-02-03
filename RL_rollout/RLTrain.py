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


import json,os

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
    DEFAULTS={"policy":"MlpPolicy","algorithm":"PPO","environment":"WurtzReact-v1","steps": 500,"dir":"<DEFAULT>","seed":None}
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
        """Returns a string with command line arguments corresponding to the options
            Outputs:
                out (str) - a single string of space-separated command line arguments
        """
        out=""
        for key in self.__dict__:
            line=key+"="+str(self.__dict__[key])
            out+=line+" "
        return out[:-1]
    
    def apply(self,args):
        """Takes in a tuple of command line arguments and turns them into options
        Inputs:
            args (tuple<str>) - Your command line arguments
        
        """
        kwargs = dict()
        for arg in args:
            try:
                key,val=arg.split("=")
                kwargs[key]=self.cmd_cast(val)
            except:pass
        self.__dict__.update(kwargs)
    def cmd_cast(self,x0):
        """Casting from a string to other datatypes
            Inputs
                x0 (string) - A string which could represent an int or float or boolean value
            Outputs
                x (?) - The best-fitting cast for x0
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
        """Depricated: Takes files formatted in the __str__ format and turns them into a set of options
        Instead of using this, consider using save() and load() functions. 
        """
        kwargs = dict()
        with open(fn,"r") as f:
          for line in f:
            line=line.strip()
            split = line.split("\t")
            key,val = split[0].strip(),split[-1].strip()
            try:
                kwargs[key]=self.cmd_cast(val)
            except:pass
        self.__dict__.update(kwargs)
        
    def save(self,fn):
        """Saves the options in json format
        Inputs:
            fn (str) - The file destination for your output file (.json is not appended automatically)
        Outputs:
            A plain text json file
        """
        with open(fn,"w") as f:
            json.dump(self.__dict__, f)
            
    def load(self,fn):
        """Saves  options stored in json format
        Inputs:
            fn (str) - The file source (.json is not appended automatically)
        """
        with open(fn,"r") as f:
            kwargs = json.load(f)
        self.__dict__.update(kwargs)




from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import gym
#from wandb.integration.sb3 import WandbCallback
#import wandb
import sys
sys.path.append('../')
import chemistrylab



ALGO={"PPO":PPO,"A2C":A2C}


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
    op.save(op.dir+"/settings.json")
        
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
    #train the model
    model.learn(
    total_timesteps=op.steps,
    log_interval=100,
    )
    
    model.save(op.dir+"\\model")

    #Clean up the logging
    sys.stdout = old_stdout
    log_file.close()