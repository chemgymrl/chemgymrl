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

from Heuristic_Policies import *


    

HEURISTICS = {
                "WRH":WurtzReactHeuristic,"FR2H":FictReact2Heuristic,
                "WDH":WurtzDistillHeuristic,"WEH":WurtzExtractHeuristic
             }


def salt_check(env):
    return any([(mat in vessel.material_dict) and (vessel.material_dict[mat].mol>1e-3)
                for vessel in env.shelf[:3] for mat in ["Na","Cl","NaCl"]])


if __name__=="__main__":
    #get options from settings.txt
    print(sys.argv[1:])
    
    op=Opt()
    #Load settings from a file but if no setting exist the user may be specifying a heuristic policy
    try:
        op.load(sys.argv[1]+"/settings.json")
        
    except:
        os.makedirs(sys.argv[1], exist_ok=True)

    
    #Allow user to change settings (like steps)
    op.apply(sys.argv[1:])
    print(op)
    
    #make the environment
    env = gym.make(op.environment)
    print("The action space is", env.action_space)
    print("The observation space is", env.observation_space)
    
    
    #choose algorithm
    if op.algorithm in HEURISTICS:
        model = HEURISTICS[op.algorithm](env,level=op.__dict__.get("level",1))
    elif not "--best" in sys.argv:
        #Load the model
        model = ALGO[op.algorithm].load(sys.argv[1]+"/model")
        print("Last:",model)
    else:
        model = ALGO[op.algorithm].load(sys.argv[1]+"/best_model")
        print("Best:",model)
    
    rollout = {"InState":[],"Action":[],"Reward":[],"OutState":[],"Done":[],"Info":[],"Step":[]}
    obs,*_=env.reset()
    print(obs.shape)
    i=0
    for x in range(op.steps):
        #testing taking actions
        act,_=model.predict(obs)
        newobs,rew,done,info,*_ = env.step(act)
        rollout["InState"]+=[obs]
        rollout["Action"]+=[act]
        rollout["Reward"]+=[rew]
        rollout["OutState"]+=[newobs]
        rollout["Done"]+=[done]
        if "Distill" in op.environment:
            info=dict(NaCl=salt_check(env))
        rollout["Info"]+=[info]
        rollout["Step"]+=[i]
        i+=1
        obs=newobs
        if done:
            i=0
            obs,*_ = env.reset()
            
    table = pd.DataFrame.from_dict(rollout, orient="columns")#,columns = ["S","A","R","S","D","I"])
    
    if not "--best" in sys.argv:
        table.to_pickle(sys.argv[1]+"/rollout")
    else:
        table.to_pickle(sys.argv[1]+"/best_rollout")
    #read with pd.read_pickle(file_name)
    print(table)
    print("Done")
            
            