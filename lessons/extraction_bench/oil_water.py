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
oil_env = [env_spec.id for env_spec in all_envs if 'Oil' in env_spec.id][0]
env = gym.make(env_ids[select_env])
# env = gym.make(oil_env)
render_mode = "human"

action_set = ['Draining from ExV to Beaker1', 'Mix ExV', "Mix B1", "Mix B2", "Pour from B1 to ExV", "Pour from B1 to B2",
              'Pour from ExV to B2', 'Add oil, pour from Oil Vessel to ExV', 'wait', 'Done']

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
    print('======================================================')
    for index, action_desc in enumerate(action_set):
        print(f'{index}: {action_desc}')
    print('Please enter an action and an action multiplier')
    for i in range(2):
        message = 'Action'
        if i == 1:
            message = 'Action Multiplier:'
        action[i] = int(input(f'{message}: '))
    state, reward, done, _ = env.step(action)
    total_reward += reward
