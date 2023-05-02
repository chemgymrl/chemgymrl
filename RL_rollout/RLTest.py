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

    def __init__(self,env,level=1):
        self.env=env
        self.level=level
        self.step=0

    def predict(self,observation):
        """
        Get an action from an observation based off of heurstics

        :param observation: (np.ndarray) the input observation

        :return: (np.ndarray, []) the model's action and an empty array (for baselines compatability)
        """
        raise NotImplementedError

class WurtzReactHeuristic(Heuristic):
    def predict(self,observation):
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
    
class FictReact2Heuristic():
    def predict(self,observation,a=0.69749509,b=1.0,thresh=0.0575):
        t = np.argmax(observation[-4:])
        #targs = [E F G I]
        actions=np.array([
        #T V A B D F G
        [1,0,1,1,0,0,0],#A+B -> E
        [1,0,1,0,1,1,1],#A+D -> F
        [1,0,0,b,1,0,1],#B+D -> G
        [1,0,0,0,0,1,1],#F+G -> I
        ],dtype=np.float32)
        #making I is a special case
        if t==3:
            marker=observation[:100].mean()
            if marker<0.01:
                #dump in a little bit of A so it's all used up
                return np.array([1,0,a,0,1,1,1]),[]# make F
            elif marker>thresh:
                #Just keep the temps up
                return np.array([0.9,0,0,0,0,0,0]),[]
            else:
                #Dump in some B
                return actions[2],[]# make G
        else: 
            return actions[t],[]
        
class FictReact1Heuristic():
    def predict(self,observation,a=0.91378666,b=0.92011728,thresh=0.093):
        t = np.argmax(observation[-5:])
        #targs = [E F G H I]
        #print("EFGHI"[t],t)
        actions=np.array([
        #T V A B C D F G H
        [1,1,1,1,1,0,0,0,0],#A+B+C -> E
        [1,1,1,0,0,1,0,0,0],#A+D -> F
        [1,1,0,1,0,1,0,1,0],#B+D -> G
        [1,1,0,0,1,1,0,0,0], #C+D -> H
        [1,1,0,0,1,0,0,0,0]# F+G+C -> I
        ],dtype=np.float32)
        #making I is a special case
        if t==4:
            marker=observation[:100].mean()
            #print(marker)
            if marker<0.01:
                #dump in a little bit of A+B and D so it's all used up
                return np.array([1,1,a,b,0,1,1,1,0]),[]# make F
            #Const is set to a value that gives the optimal time to add B
            elif marker>thresh:
                #Just keep the temps up
                return np.array([1,1,0,0,0,0.0,0,0,0]),[]
            else:
                #Dump in some C
                return actions[t],[]# make G
        else: 
            return actions[t],[]
    
class WurtzDistillHeuristic(Heuristic):
    level_2 = "0909090940"
    level_3 = ["0909090940","0909090929090909090940"]
    def predict(self,observation):
        if self.level==2:
            pol= WurtzDistillHeuristic.level_2
        elif self.level==3:
            pol= WurtzDistillHeuristic.level_3[salt_check(self.env)]
            
        if self.level>1:
            act,param = pol[2*self.step:2*self.step+2]
        else:
            act=np.random.choice([0]*3+[4])
            param=9
        
        self.step+=1
        if int(act)==4:
            self.step=0
        return np.array([int(act),int(param)]),[]
    
class WurtzExtractHeuristic(Heuristic):
    level_2 = "501474010404040480"
    level_3 = "5074700004040404647472040404040480"
    
    def mix_check(self,o):
        if self.step==0:
            self.step+=1
            return 5*5
        
        if self.step==49:
            self.step=0
            return 8*5
        #Drain if the dodecane is near the bottom
        elif o[0,0:14].mean()>0.72:
            self.step+=1
            return 4
        elif o[0,0:11].mean()>0.72:
            self.step+=1
            return 3
        elif o[0,0:8].mean()>0.72:
            self.step+=1
            return 2
        elif o[0,0:5].mean()>0.72:
            self.step+=1
            return 1
        elif o[0,0:2].mean()>0.72:
            self.step+=1
            return 0
            
        #if not mix then wait one step
        else:
            self.step+=1
            if self.step%2==0:
                return 1*5+2
            else:
                return 7*5

    def predict(self,observation):
        if self.level==1:
            return self.mix_check(observation),[]
        if self.level==2:
            pol= WurtzExtractHeuristic.level_2
        elif self.level==3:
            pol= WurtzExtractHeuristic.level_3
        
        act,param = pol[2*self.step:2*self.step+2]
        
        self.step+=1
        if int(act)==8:
            self.step=0
        return int(act)*5+int(param),[]


HEURISTICS = {"WRH":WurtzReactHeuristic,"FR2H":FictReact2Heuristic,"WDH":WurtzDistillHeuristic,"WEH":WurtzExtractHeuristic,
             "FR1H":FictReact1Heuristic
             
             }


def salt_check(env):
    return any([(mat in vessel.material_dict) and (vessel.material_dict[mat].mol>1e-3)
                for vessel in env.vessels for mat in ["Na","Cl","NaCl"]])


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
        if "Distill" in op.environment:
            info=dict(NaCl=salt_check(env))
        rollout["Info"]+=[info]
        rollout["Step"]+=[i]
        i+=1
        obs=newobs
        if done:
            i=0
            obs = env.reset()
            
    table = pd.DataFrame.from_dict(rollout, orient="columns")#,columns = ["S","A","R","S","D","I"])
    
    if not "--best" in sys.argv:
        table.to_pickle(sys.argv[1]+"/rollout")
    else:
        table.to_pickle(sys.argv[1]+"/best_rollout")
    #read with pd.read_pickle(file_name)
    print(table)
    print("Done")
            
            