import gym
import chemistrylab
import numpy as np

# Random agent just chooses random action
def random_policy(env):
    return env.action_space.sample()

def coded_policy(s):
    t = np.argmax(s[-7:])
    targs = {0: "dodecane", 1: "5-methylundecane", 2: "4-ethyldecane", 3: "5,6-dimethyldecane", 4: "4-ethyl-5-methylnonane", 5: "4,5-diethyloctane", 6: "NaCl"}
    target = targs[t]
    a = np.zeros(6, dtype=np.float32)
    a[0] += 1
    if target == "dodecane":
        a[2] += 1.0
        a[5] += 1.0

    elif target == "5-methylundecane":
        a[2] += 1.0
        a[3] += 1.0
        a[5] += 1.0

    elif target == "4-ethyldecane":
        a[2] += 1.0
        a[4] += 1.0
        a[5] += 1.0

    elif target == "5,6-dimethyldecane":
        a[3] += 1.0
        a[5] += 1.0

    elif target == "4-ethyl-5-methylnonane":
        a[3] += 1.0
        a[4] += 1.0
        a[5] += 1.0

    elif target == "4,5-diethyloctane":
        a[4] += 1.0
        a[5] += 1.0

    elif target == "NaCl":
        a[2] += 1.0
        a[3] += 1.0
        a[4] += 1.0
        a[5] += 1.0

    return a

env = gym.make("GenWurtzReact-v1")

R0 = np.zeros(7, dtype=np.float32)
R1 = np.zeros(7, dtype=np.float32)
c0 = np.zeros(7, dtype=np.int32)
c1 = np.zeros(7, dtype=np.int32)

for i in range(1000):
    s = env.reset()
    t = np.argmax(s[-7:])
    c0[t] += 1
    done = False
    while not done:
        a = random_policy(env)
        s, r, done, _ = env.step(a)
        R0[t] += r

    s = env.reset()
    t = np.argmax(s[-7:])
    c1[t] += 1
    done = False
    while not done:
        a = coded_policy(s)
        s, r, done, _ = env.step(a)
        R1[t] += r

print((R0 / c0).sum(), R0 / c0)
print((R1 / c1).sum(), R1 / c1)
