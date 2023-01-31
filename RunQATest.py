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

Runs the QATest script on all registered chemgym environments

:title: RunQATest.py

:author: Kyle Sprague

:history: 22-01-2023
"""



import os
import sys

sys.path.append('./chemistrylab')
import gym
import chemistrylab
import numpy as np
from gym import envs

all_envs = envs.registry.all()
def ischemgym(x):
    if "Distill" in x:return True
    if "Extract" in x: return True
    if "React" in x: return True
    return False

env_ids = [env_spec.id for env_spec in all_envs if ischemgym(env_spec.id)]
print(env_ids)


#os.remove("UnitTest.py")
with open(".\\message.log","w") as f:
    pass

for id in env_ids:
    print("-"*60+"\n"+id,end=":")
    x = os.system("python QATest.py %s 200"%id)
    if x==1:
        with open("./message.log","r") as f:
            txt=""
            for line in f:
                txt=line
        print("Failed:\n"+line)
    elif x==0:
        print("Passed")