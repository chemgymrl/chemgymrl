from Heuristic import Heuristic
import numpy as np

class WurtzReactHeuristic(Heuristic):
    def predict(self,observation):
        t = np.argmax(observation[-7:])
        #targs = {0: "dodecane", 1: "5-methylundecane", 2: "4-ethyldecane", 3: "5,6-dimethyldecane", 4: "4-ethyl-5-methylnonane", 5: "4,5-diethyloctane", 6: "NaCl"}
        actions=np.array([
        [1,1,0,0,1],#dodecane
        [1,1,1,0,1],#5-methylundecane
        [1,1,0,1,1],#4-ethyldecane
        [1,0,1,0,1],#5,6-dimethyldecane
        [1,0,1,1,1],#4-ethyl-5-methylnonane
        [1,0,0,1,1],#4,5-diethyloctane
        [1,1,1,1,1],#NaCl
        ],dtype=np.float32)
        return actions[t],[]
    

class FictReact2Heuristic(Heuristic):
    def predict(self,observation,a=0.91378666,b=0.92011728,thresh=0.093):
        t = np.argmax(observation[-5:])
        #targs = [E F G H I]
        #print("EFGHI"[t],t)
        actions=np.array([
        #T A B C D
        [1,1,1,1,0],#A+B+C -> E
        [1,1,0,0,1],#A+D -> F
        [1,0,1,0,1],#B+D -> G
        [1,0,0,1,1], #C+D -> H
        [1,0,0,1,0]# F+G+C -> I
        ],dtype=np.float32)
        #making I is a special case
        if t==4:
            marker=observation[:100].mean()
            #print(marker)
            if marker<0.01:
                #dump in a little bit of A+B and D so it's all used up
                return np.array([1,a,b,0,1]),[]# make F
            #Const is set to a value that gives the optimal time to add B
            elif marker>thresh:
                #Just keep the temps up
                return np.array([1,0,0,0,0]),[]
            else:
                #Dump in some C
                return actions[t],[]# make G
        else: 
            return actions[t],[]