from Heuristic import Heuristic
import numpy as np


def salt_check(env):
    return any([(mat in vessel.material_dict) and (vessel.material_dict[mat].mol>1e-3)
                for vessel in env.shelf[:3] for mat in ["Na","Cl","NaCl"]])

class WurtzDistillHeuristic(Heuristic):
    level_2 = "0909090930"
    level_3 = ["0909090930","090909090929090909090930"]
    def predict(self,observation):
        if self.level==2:
            pol= WurtzDistillHeuristic.level_2
        elif self.level==3:
            pol= WurtzDistillHeuristic.level_3[salt_check(self.env)]
            
        if self.level>1:
            act,param = pol[2*self.step:2*self.step+2]
        else:
            act=np.random.choice([0]*3+[3])
            param=9 if act == 0 else 0
        
        self.step+=1
        if int(act)==3:
            self.step=0
        return int(act)*10+int(param),[]
    
    