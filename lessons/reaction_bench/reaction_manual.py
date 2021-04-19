"""
This file is part of ChemGymRL.

ChemGymRL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ChemGymRL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ChemGymRL.  If not, see <https://www.gnu.org/licenses/>.
"""
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
render_mode = "full"

action_set = ['Temperature', 'Volume'] + env.reaction.reactants

assert len(action_set) == env.action_space.shape[0]
print(env.vessels.get_material_dict())
env.reaction.solver = "RK45"
done = False
state = env.reset()
total_reward = 0
while not done:
    # print(state)
    env.render(mode=render_mode)
    action = np.zeros(env.action_space.shape[0])
    print('--------------------')
    print(env.reaction.cur_in_hand)
    print("solute_dict")
    print(env.vessels.get_solute_dict())
    for i, a in enumerate(action_set):
        action[i] = float(input(f'{a}: '))
    state, reward, done, _ = env.step(action)
    total_reward += reward
    print("materials")
    print(env.vessels.get_material_dict())
    print(f'reward: {reward}')
    print(f'total_reward: {total_reward}')