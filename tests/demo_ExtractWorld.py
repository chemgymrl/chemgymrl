import gym
import chemistrylab
import numpy as np
from time import sleep
import time

env = gym.make('ExtractWorld_0-v1')
s = env.reset()
env.render()
s, r, d, _ = env.step(np.array([3,2]))
env.render()
_ = input('Press ENTER to continue')

d = False
i = 0

reward = 0.0
while not d:
    # Select random actions
    a = env.action_space.sample()
    b = env.action_space
    print("The value is", b.nvec[0])
#    if int(a[0]) == 8:  # don't end it, even if randomly sampled done
#        continue
#    start = time.time()
    s, r, d, _ = env.step(a)
#    s, r, d, _ = env.step(np.array([0.5,0.5], dtype=np.float32))
#    end = time.time()
#    print(end - start)
    
    reward += r
    env.render()
    sleep(1)  # Wait 1 seconds before continuing
    i += 1
    print(i, a, r, d)


#from extractworldgym.extractworld_v1_engine import ExtractWorldEnv
#from chemistrylab import *
#
## initialize extraction vessel
#extraction_vessel = vessel.Vessel(label='extraction_vessel',
#                                  )
#
## initialize materials
#H2O = material.H2O()
#Na = material.Na()
#Cl = material.Cl()
#Na.set_charge(1.0)
#Na.set_solute_flag(True)
#Cl.set_charge(-1.0)
#Cl.set_solute_flag(True)
#
## material_dict
#material_dict = {H2O.get_name(): [H2O, 0.5],
#                 Na.get_name(): [Na, 1.0],
#                 Cl.get_name(): [Cl, 1.0]
#                 }
#solute_dict = {Na.get_name(): {H2O.get_name(): 1.0},
#               Cl.get_name(): {H2O.get_name(): 1.0},
#               }
#v = util.convert_material_dict_to_volume(material_dict,
#                                    # the density of solution does not affect the original volume of solvent
#                                    )
#
#event_1 = ['update material dict', material_dict]
#event_2 = ['update solute dict', solute_dict]
#event_3 = ['fully mix']
#
#extraction_vessel.push_event_to_queue(events=None, feedback=[event_1, event_2, event_3], dt=0)
