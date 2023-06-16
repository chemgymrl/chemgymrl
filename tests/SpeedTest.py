import os
import sys

# set output to a log file called message.log
old_stdout = sys.stdout
log_file = open(os.path.dirname(__file__)+"/message.log","w")
sys.stdout = log_file



sys.path.append('..')

sys.path.append("\\".join(os.path.dirname(__file__).split("\\")[:-1]))

import os
print("\\".join(os.path.dirname(__file__).split("\\")[:-1]))
sys.stdout.flush()

import gymnasium as gym
import chemistrylab
import numpy as np
from matplotlib import pyplot as plt
from chemistrylab import material



#time.time() isn't the very best for timing but I don't care
import time

def chemgym_filter(x):
    y=[]
    for env in x:
        if ("React" in env or "Extract" in env or "Distill" in env) and not "Discrete" in env:
                y+=[env]
    return y

def run_env(env_id):
    env=gym.make(env_id)
    env.metadata["render_modes"] = ["rgb_array"]
    _=env.reset()
    #
    #env.reaction.newton_steps=100
    if "React" in env_id:
        env.default_events[0].parameter[0].solver="newton"
        env.default_events[0].parameter[0].newton_steps=1000
    t0 = time.perf_counter()
    for x in range(1000):
        o,r,d,*_=env.step(env.action_space.sample())
        #env.render()
        if d:
            env.reset()
    t= time.perf_counter()
    return t-t0

names = chemgym_filter([a for a in gym.envs.registry])

try:
    [run_env(env_id) for env_id in names]
except:
    pass

for env_id in names:
    try:
        dt=run_env(env_id)
        print(env_id,int(dt*1000),"ms")
        sys.stdout.flush()
    except Exception as e:
        print(env_id,e)
    
sys.stdout = old_stdout

log_file.close()
