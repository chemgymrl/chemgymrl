'''
LabManager Environment

:title: manager_v1.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-08-07
'''

import gym
import sys

sys.path.append("../../")
from chemistrylab.manager.manager_v1_engine import LabManagerEnv

def get_environments():
    '''
    Function to acquire all of the environments available to the lab manager.

    Parameters
    ---------------
    `n_steps` : `int` (default=`100`)
        The number of time steps to be taken during each action.
    `environments` : `list` (default=`None`)
        A list of environments available to the lab manager.
    `out_vessel_path` : `str` (default=`None`)
        A string indicating the path to a vessel intended to be loaded into this module.

    Returns
    ---------------
    `envs_list` : `list`
        A list of all the registered environments that
        are allowed to be accessed by the Lab Manager.

    Raises
    ---------------
    None
    '''

    # create all the environments that are to be made available to the lab manager
    r_env = gym.make('WurtzReact-v0')
    e_env = gym.make('WurtzExtract-v1')
    d_env = gym.make('Distillation-v0')

    # return a list of all the available environments
    envs_list = [r_env, e_env, d_env]

    return envs_list

class LabManager(LabManagerEnv):
    '''
    Class object to define the Lab Manager environment and
    the environments made available to the Lab Manager.
    '''

    def __init__(self):
        super(LabManager, self).__init__(
            environments=get_environments(),
            out_vessel_path=""
        )