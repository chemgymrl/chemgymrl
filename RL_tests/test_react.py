import gym
import numpy as np
import chemistrylab
from stable_baselines3 import PPO

env = gym.make("FictReact-v2")

model = PPO.load("PPO_React.zip")

R = np.zeros(4, dtype=np.float32)
c = np.zeros(4, dtype=np.int32)

for i in range(50):
    s = env.reset()
    t = np.argmax(s[-4:])
    c[t] += 1
    done = False
    while not done:
        a, _s = model.predict(s, deterministic=True)
        s, r, done, _ = env.step(a)
        R[t] += r

print((R / c).mean(), R / c)
