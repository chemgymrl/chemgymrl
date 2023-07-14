from chemistrylab.lab.heuristics.Heuristic import Policy
import numpy as np


def salt_check(env):
    return any([(mat in vessel.material_dict) and (vessel.material_dict[mat].mol>1e-3)
                for vessel in env.shelf[:3] for mat in ["[Na+]","[Cl-]","[Na+].[Cl-]"]])

class WurtzDistillHeuristic(Policy):
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
        return int(act)*10+int(param)
    

class GenWurtzDistillHeuristic(Policy):
    def __init__(
        self, 
        heat_act, 
        b1_act, 
        b2_act, 
        done_act, 
        layer_pos, # should be three length 2 tuples
        salt_color = 911, 
        solvent_color = 951, 
        ):

        self.heat_act = heat_act
        self.b1_act = b1_act
        self.b2_act = b2_act
        self.done_act = done_act 
        self.layer_pos = layer_pos
        self.salt_color = salt_color
        self.solvent_color = solvent_color

    def get_layer_info(self,obs, layer_start, layer_end):
        layer_info  = obs[layer_start:layer_end]
        eps=1e-3
        coords = np.arange(layer_info.shape[0]+1)
        y = np.sort(layer_info)
        break_points = np.ones(y.shape[0]+1,dtype=np.bool_)
        break_points[1:-1]=y[1:]-y[:-1] > eps
        indices = coords[break_points]
        clusters = tuple(y[indices[i]:indices[i+1]] for i in range(indices.shape[0]-1))
        cluster_info = {int(x.mean()*1000+0.5):x.shape[0] for x in clusters}

        return cluster_info

    def predict(self, observaiton):
        vessel_1_contents = self.get_layer_info(observaiton, *self.layer_pos[0])
        vessel_2_contents = self.get_layer_info(observaiton, *self.layer_pos[1])
        vessel_3_contents = self.get_layer_info(observaiton, *self.layer_pos[2])


        if self.solvent_color in vessel_1_contents:
            return self.heat_act

        if self.solvent_color in vessel_2_contents:
            return self.b2_act

        if self.salt_color in vessel_1_contents and len(vessel_1_contents)>2:
            return self.heat_act

        return self.done_act