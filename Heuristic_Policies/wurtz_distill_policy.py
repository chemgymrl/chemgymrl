import gym
import chemistrylab
import numpy as np

# Random agent just chooses random action
def random_policy(env):
    return env.action_space.sample()

def coded_policy(step):
    if step <= 66:
        return [0, 9]
    else:
        return [4, 9]

env = gym.make("WurtzDistill-v1")

R0 = np.zeros(7, dtype=np.float32)
R1 = np.zeros(7, dtype=np.float32)
c0 = np.zeros(7, dtype=np.int32)
c1 = np.zeros(7, dtype=np.int32)

for i in range(5):
    s = env.reset()
    t = np.argmax(s[0, -7:])
    c0[t] += 1
    done = False
    while not done:
        a = random_policy(env)
        s, r, done, _ = env.step(a)
        R0[t] += r

    s = env.reset()
    t = np.argmax(s[0, -7:])
    c1[t] += 1
    step = 0
    done = False
    while not done:
        a = coded_policy(step)
        s, r, done, _ = env.step(a)
        R1[t] += r
        step += 1

print((R0 / c0).sum(), R0 / c0)
print((R1 / c1).sum(), R1 / c1)
