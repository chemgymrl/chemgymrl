import gym
import numpy as np
import chemistrylab
from chemistrylab.reaction_bench.reaction_bench_v1 import FictReact_v1, FictReact_v2
import time

env = FictReact_v2(target_material="I")
s = env.reset()
'''
for i in range(5):
    s, r, d, _ = env.step([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

for i in range(15):
    s, r, d, _ = env.step([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

print("Heuristic")
print("E: " + str(env.vessels._material_dict["E"][1]))
print("I: " + str(env.vessels._material_dict["I"][1]))
print(r)
print("")
'''

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
print("E: " + str(env.vessels._material_dict["E"][1]))
print("I: " + str(env.vessels._material_dict["I"][1]))
print(re/100)
