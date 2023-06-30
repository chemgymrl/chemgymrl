from Heuristic import Heuristic



class WurtzExtractHeuristic(Heuristic):
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
    
    
    
    
class WaterOilHeuristic(Heuristic):
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
        return int(act)*5+int(param),[]
    
    
    