from time import sleep
import gym
import chemistrylab
from stable_baselines3 import DQN, PPO, A2C

env = gym.make('GenWurtzReact-v1')

model = PPO.load("PPO_React")

'''
R = 0
for i in range(10):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        R += reward

print(R/10)
'''

obs = env.reset()
print(obs[-7:])
env.render()
input('Press Enter')
env.render()
input('Press Enter')
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    sleep(0.2)

print(reward)
