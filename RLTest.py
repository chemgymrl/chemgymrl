from RLTrain import *
import pandas as pd
print(ALGO)



class WurtzReactHeuristic():
  def predict(s):
    t = np.argmax(s[-7:])
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
    return actions[t],{}


HEURISTICS = {"WRH":WurtzReactHeuristic}


if __name__=="__main__":
    #get options from settings.txt
    print(sys.argv[1:])
    
    op=Opt()
    #Load settings from a file but if no setting exist the user may be specifying a heuristic policy
    try:
        op.from_file(sys.argv[1]+"/settings.txt")
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
        model = ALGO[op.algorithm].load(sys.argv[1]+"/best_model.zip")
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
            
            