from RLTrain import *
import pandas as pd
print(ALGO)


if __name__=="__main__":
    #get options from settings.txt
    print(sys.argv[1:])
    op=Opt()
    op.from_file(sys.argv[1]+"/settings.txt")
    print(op)
    
    #Load the model
    model = ALGO[op.algorithm].load(sys.argv[1]+"/best_model.zip")
    print(model)
    
    
    #make the environment
    env = gym.make(op.environment)
    print("The action space is", env.action_space)
    print("The observation space is", env.observation_space)
    
    
    if len(sys.argv)>2:op.steps=int(sys.argv[2])
    
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
            
            