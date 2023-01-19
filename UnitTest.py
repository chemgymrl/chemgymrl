import os
import sys

old_stdout = sys.stdout

log_file = open(".\message.log","a")

sys.stdout = log_file


import gym
import chemistrylab
import numpy as np
from gym import envs


id = sys.argv[1]
    
timesteps = int(sys.argv[2])
    
print("-"*60+"\n"+id+" TEST START:")

print("Checking case for did not halt. . .")
sys.stdout = old_stdout
log_file.close()
log_file = open(".\message.log","a")
sys.stdout = log_file


try:
    env=gym.make(id)
    obstest = env.observation_space.sample()
    obs0 = env.reset()
    
    #testing if the observation matches the observation space
    _ = obstest*obs0
    
    for x in range(timesteps):
        #testing taking actions
        act = env.action_space.sample()
        obs,rew,done,_ = env.step(act)
        
        #testing if the observation matches the observation space
        _ = obstest*obs
        
        if done:
            obs0 = env.reset()
            #duplicate test forr observation space
            _ = obstest*obs0
    
    
    #print("Passed")
except Exception as e:
    #print("failed")
    print("-"*60+"\n"+id+" TEST FAILED:")
    print(e)
    
    sys.stdout = old_stdout
    log_file.close()
    1/0
        
        
print("-"*60+"\n"+id+" TEST PASSED",end=":")

sys.stdout = old_stdout

log_file.close()