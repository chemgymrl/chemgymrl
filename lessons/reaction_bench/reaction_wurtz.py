import sys
sys.path.append('../../')
sys.path.append('../../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np
from gym import envs

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if 'React' in env_spec.id]
print(env_ids)
select_env = int(input(f"enter a number to chose which environment you want to run (0-{len(env_ids) - 1}): "))
env = gym.make(env_ids[select_env])
render_mode = "human"

action_set = ['Temperature', 'Volume', "1-chlorohexane", "2-chlorohexane", "3-chlorohexane", "Na"]

assert len(action_set) == env.action_space.shape[0]

done = False
state = env.reset()
total_reward = 0
round = 0

while not done:
    # print(state)
    env.render(mode=render_mode)
    action = np.zeros(env.action_space.shape[0])
    if select_env == 1:
        if round == 0:
            action[0] = 1
            action[1] = 1
            action[2] = 1
            action[-1] = 1
        else:
            action[0] = 1
            action[1] = 1
    elif select_env == 0:
        if round == 0:
            action = np.ones(env.action_space.shape[0])
            action[0] = 1
            action[1] = 0
            # action[0] = 1
            # action[1] = 1
            # action[2] = 1
            # action[-1] = 1
        else:
            # action[0] = 0.5
            # action[1] = 0
            action[0] = 1
            action[1] = 0
    print('--------------------')
    print(round)
    # for i, a in enumerate(action_set):
    #     action[i] = float(input(f'{a}: '))
    state, reward, done, _ = env.step(action)
    total_reward += reward
    print(f'reward: {reward}')
    print(f'total_reward: {total_reward}')
    round += 1
    if done:
        wait = input("PRESS ENTER TO EXIT")
