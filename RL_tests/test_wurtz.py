import gym
import numpy as np
import chemistrylab
from chemistrylab.reaction_bench.reaction_bench_v1 import GeneralWurtzReact_v1
import time

env = GeneralWurtzReact_v1(target_material="dodecane")
s = env.reset()

re = 0
count = 0
tic = time.time()
for ep in range(100):
    count += 1
    s = env.reset()
    for i in range(20):
        s, r, d, _ = env.step(env.action_space.sample())
    re += r
    toc = time.time()
    print("Time " + str(ep+1) + ": " + str((toc - tic)/count))

print("\nRandom")
print(re/100)
