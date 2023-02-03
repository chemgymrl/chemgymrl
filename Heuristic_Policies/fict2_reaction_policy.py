import gym
import chemistrylab
import numpy as np

# Random agent just chooses random action
def random_policy(env):
    return env.action_space.sample()

def coded_policy(s, step):
    t = np.argmax(s[-4:])
    targs = {0: "E", 1: "F", 2: "G", 3: "I"}
    target = targs[t]
    a = np.zeros(7, dtype=np.float32)
    a[0] += 1
    if target == "E":
        a[2] += 1.0
        a[3] += 1.0

    elif target == "F":
        a[2] += 1.0
        a[4] += 1.0

    elif target == "G":
        a[3] += 1.0
        a[4] += 1.0

    elif target == "I":
        if step < 10:
            a[2] += 1.0
            a[4] += 1.0
        else:
            a[3] += 1.0
            a[4] += 1.0

    return a

env = gym.make("FictReact-v2")

R0 = np.zeros(4, dtype=np.float32)
R1 = np.zeros(4, dtype=np.float32)
c0 = np.zeros(4, dtype=np.int32)
c1 = np.zeros(4, dtype=np.int32)

for i in range(50):
    s = env.reset()
    t = np.argmax(s[-4:])
    c0[t] += 1
    done = False
    while not done:
        a = random_policy(env)
        s, r, done, _ = env.step(a)
        R0[t] += r

    s = env.reset()
    step = -1
    t = np.argmax(s[-4:])
    c1[t] += 1
    done = False
    while not done:
        step += 1
        a = coded_policy(s, step)
        s, r, done, _ = env.step(a)
        R1[t] += r

print((R0 / c0).mean(), R0 / c0)
print((R1 / c1).mean(), R1 / c1)
