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

Module to run a gym with a random agent to ensure the code runs without error.
It calls gym.make() on the specified gym environment, runs it for a set number of steps
using a random agent which samples from the action space, then logs the output. Any errors
encountered are also logged.

Usage from a command line works as follows:
>>python QATest.py <Environment> <steps>

Example call:
>>python QATest.py GenWurtzExtract-v1 200



:title: QATest.py

:author: Kyle Sprague

:history: 22-01-2023
"""


import os
import sys

# set output to a log file called message.log
old_stdout = sys.stdout
log_file = open(".\message.log","a")
sys.stdout = log_file


import gym
import chemistrylab
import numpy as np
from gym import envs

#get the gym environemnt id
id = sys.argv[1]
  
#get the number of timesteps
timesteps = int(sys.argv[2])
    
print("-"*60+"\n"+id+" TEST START:")

print("Checking case for did not halt. . .")

# Write to the stdout file and reopen it. This is so if
# the script terminates unexpectedly there is some output.
sys.stdout = old_stdout
log_file.close()
log_file = open(".\message.log","a")
sys.stdout = log_file


try:
    #create the gym environment
    env=gym.make(id)
    obstest = env.observation_space.sample()
    obs0 = env.reset()
    
    #testing if the observation matches the observation space
    _ = obstest*obs0
    
    for x in range(timesteps):
        #testing taking actions
        act = env.action_space.sample()
        obs,rew,done,_ = env.step(act)
        
        #testing if the observation matches the observation space
        _ = obstest*obs
        
        if done:
            obs0 = env.reset()
            #duplicate test forr observation space
            _ = obstest*obs0
    
    
    #print("Passed")
except Exception as e:
    #print("failed")
    print("-"*60+"\n"+id+" TEST FAILED:")
    print(e)
    
    sys.stdout = old_stdout
    log_file.close()
    1/0
        
        
print("-"*60+"\n"+id+" TEST PASSED",end=":")

sys.stdout = old_stdout

log_file.close()