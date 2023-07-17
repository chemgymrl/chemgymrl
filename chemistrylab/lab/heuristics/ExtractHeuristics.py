from chemistrylab.lab.heuristics.Heuristic import Policy
import numpy as np

class WurtzExtractHeuristic(Policy):
    level_2 = "501474020404040480"
    level_3 = "5074740204040404647474040404040480"
    
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
            return self.mix_check(observation)
        if self.level==2:
            pol= WurtzExtractHeuristic.level_2
        elif self.level==3:
            pol= WurtzExtractHeuristic.level_3
        
        act,param = pol[2*self.step:2*self.step+2]
        
        self.step+=1
        if int(act)==8:
            self.step=0
        return int(act)*5+int(param)
    
class GenWurtzExtractHeuristic(Policy):
    def __init__(
        self, 
        drain_act, 
        polar_act, 
        non_polar_act, 
        mix_act, 
        wait_act, 
        done_act, 
        layer_pos, 
        polar_color = 955, 
        non_polar_color = 951, 
        history=6
        ):
        """
        Args:
        - drain_act: the integer action to drain
        - polar_act: the integer action to add a polar solvent
        - non_polar_act: the integer action to add a non-polar solvent
        - mix_act: the integer action to mix
        - wait_act: the integer action to wait
        - layer_pos: A (i,j) tuple where obs[i:j] gives the layer encoding of the observaiton
        """
        self.drain_act = drain_act
        self.polar_act = polar_act
        self.non_polar_act = non_polar_act
        self.mix_act = mix_act
        self.wait_act = wait_act
        self.layer_pos = layer_pos
        self.done_act = done_act

        self.polar_color = polar_color

        self.non_polar_color = non_polar_color

        self.prev=None

        self.prev_correlations = [0]*history
    
    def get_layer_info(self,obs):
        layer_info  = obs[self.layer_pos[0]:self.layer_pos[1]]

        correlation = np.mean(abs(layer_info[1:]-layer_info[:-1])<1e-3)*2-1

        self.prev_correlations = self.prev_correlations[1:]+[correlation]

        correlation=min(self.prev_correlations)
        eps=1e-3

        coords = np.arange(layer_info.shape[0]+1)
        y = np.sort(layer_info)
        break_points = np.ones(y.shape[0]+1,dtype=np.bool_)
        break_points[1:-1]=y[1:]-y[:-1] > eps
        indices = coords[break_points]
        clusters = tuple(y[indices[i]:indices[i+1]] for i in range(indices.shape[0]-1))
        cluster_info = {int(x.mean()*1000+0.5):x.shape[0] for x in clusters}

        correlation+=len(cluster_info)/(layer_info.shape[0]-1)

        return correlation, cluster_info

    def choose_action(self,observation):
        correlation, cluster_info = self.get_layer_info(observation)

        #print(cluster_info,correlation)

        polar_amount = cluster_info.get(self.polar_color,0)

        non_polar_amount = cluster_info.get(self.non_polar_color,0)

        air_amount = cluster_info.get(1000,0)

        # Constants
        TOL = 0.955
        drained=10
        full=42

        #draining phase
        if (correlation > TOL or self.prev==self.drain_act) and min(polar_amount,non_polar_amount)>drained:
            return self.drain_act

        if air_amount<=drained:
            return self.wait_act 

        if correlation <= TOL or min(polar_amount,non_polar_amount)<=drained:
            #waiting phase
            if min(polar_amount,non_polar_amount)>=full:
                return self.wait_act 

            #pouring phase
            if polar_amount<full or polar_amount<=drained:
                return self.polar_act
            
            if non_polar_amount<full  or non_polar_amount<=drained:
                return self.non_polar_act

        return self.done_act

    def predict(self,observation):
        self.prev = self.choose_action(observation)
        return self.prev
        
    
class WaterOilHeuristic(Policy):
    level_2 = "63147474040404044480"
    level_3 = "631474740404040463147474040404044480"

    def predict(self,observation):
        if self.level==1:
            return 40
        if self.level==2:
            pol= WaterOilHeuristic.level_2
        elif self.level==3:
            pol= WaterOilHeuristic.level_3
        
        act,param = pol[2*self.step:2*self.step+2]
        
        self.step+=1
        if int(act)==8:
            self.step=0
        return int(act)*5+int(param)
    
    
    