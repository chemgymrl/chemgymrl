import sys
sys.path.append('../../')
sys.path.append('../../chemistrylab/reactions')
import gym
import chemistrylab
import numpy as np
from gym import envs

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if 'Extract' in env_spec.id]
print(env_ids)
select_env = int(input(f"enter a number to chose which environment you want to run (0-{len(env_ids) - 1}): "))
env = gym.make(env_ids[select_env])
render_mode = "human"

action_set = ['Draining from ExV to Beaker1', 'Mix ExV', "Mix B1", "Mix B2", "Pour from B1 to ExV", "Pour from B1 to B2",
              'Pour from ExV to B2', 'Add oil, pour from Oil Vessel to ExV', 'Done']
print(len(action_set))
print(env.action_space.shape)

assert 2 == env.action_space.shape[0]

done = False
state = env.reset()
total_reward = 0
reward = 0
while not done:
    # print(state)
    env.render(mode=render_mode)
    action = np.zeros(env.action_space.shape[0])
    print(f'reward: {reward}')
    print(f'total_reward: {total_reward}')
    print('--------------------')
    print(action_set)
    for i in range(2):
        action[i] = int(input(f'{i}: '))
    state, reward, done, _ = env.step(action)
    total_reward += reward
