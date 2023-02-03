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

Module to perform rollout of RL agent on a gym. Options are handled with the Opt() class.

Usage from a command line works as follows:
>>python RLTest.py <Folder> <option1 name>=<option1 value> <option2 name>=<option2 value>. . .

Example call:
>>python RLTest.py WRH algorithm=WRH steps=500

:title: RLTest.py

:author: Kyle Sprague

:history: 22-01-2023
"""



from RLTrain import *
import pandas as pd
import numpy as np
print(ALGO)



class Heuristic():
  """
  Heuristic policy for a chemgymrl bench.
  
  The predict function is implemented the same as in a stable baselines policy, so
  you should be able to use this heuristic policy in place of an RL algorithm.
  
  """
  def predict(observation):
    """
    Get an action from an observation based off of heurstics
    
    :param observation: (np.ndarray) the input observation

    :return: (np.ndarray, []) the model's action and an empty array (for baselines compatability)
    """
    raise NotImplementedError

class WurtzReactHeuristic(Heuristic):
    def predict(observation):
        t = np.argmax(observation[-7:])
        #targs = {0: "dodecane", 1: "5-methylundecane", 2: "4-ethyldecane", 3: "5,6-dimethyldecane", 4: "4-ethyl-5-methylnonane", 5: "4,5-diethyloctane", 6: "NaCl"}
        actions=np.array([
        [1,0,1,0,0,1],#dodecane
        [1,0,1,1,0,1],#5-methylundecane
        [1,0,1,0,1,1],#4-ethyldecane
        [1,0,0,1,0,1],#5,6-dimethyldecane
        [1,0,0,1,1,1],#4-ethyl-5-methylnonane
        [1,0,0,0,1,1],#4,5-diethyloctane
        [1,0,1,1,1,1],#NaCl
        ],dtype=np.float32)
        return actions[t],[]
    
class FictReact2Heuristic(Heuristic):
    def predict(observation):
        t = np.argmax(observation[-4:])
        #targs = [E F G I]
        actions=np.array([
        #T V A B D F G
        [1,0,1,1,0,0,0],#A+B -> E
        [1,0,1,0,1,1,1],#A+D -> F
        [1,0,0,1,1,0,1],#B+D -> G
        [1,0,0,0,0,1,1],#F+G -> I
        ],dtype=np.float32)
        return actions[t],[]


HEURISTICS = {"WRH":WurtzReactHeuristic,"FR2H":FictReact2Heuristic}


if __name__=="__main__":
    #get options from settings.txt
    print(sys.argv[1:])
    
    op=Opt()
    #Load settings from a file but if no setting exist the user may be specifying a heuristic policy
    try:
        op.load(sys.argv[1]+"/settings.json")
        print(op)
    except:
        os.makedirs(sys.argv[1], exist_ok=True)
    
    #Allow user to change settings (like steps)
    op.apply(sys.argv[1:])
    
    #choose algorithm
    if op.algorithm in HEURISTICS:
        model = HEURISTICS[op.algorithm]
    else:
        #Load the model
        model = ALGO[op.algorithm].load(sys.argv[1]+"/model")
        print(model)
    
    
    #make the environment
    env = gym.make(op.environment)
    print("The action space is", env.action_space)
    print("The observation space is", env.observation_space)
    
    rollout = {"InState":[],"Action":[],"Reward":[],"OutState":[],"Done":[],"Info":[],"Step":[]}
    obs=env.reset()
    i=0
    for x in range(op.steps):
        #testing taking actions
        act,_=model.predict(obs)
        newobs,rew,done,info = env.step(act)
        rollout["InState"]+=[obs]
        rollout["Action"]+=[act]
        rollout["Reward"]+=[rew]
        rollout["OutState"]+=[newobs]
        rollout["Done"]+=[done]
        rollout["Info"]+=[info]
        rollout["Step"]+=[i]
        i+=1
        obs=newobs
        if done:
            i=0
            obs = env.reset()
            
    table = pd.DataFrame.from_dict(rollout, orient="columns")#,columns = ["S","A","R","S","D","I"])
    
    table.to_pickle(sys.argv[1]+"/rollout")
    
    #read with pd.read_pickle(file_name)
    print(table)
    print("Done")
            
            