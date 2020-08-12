'''
'''

import gym

class LabManagerOperation:
    '''
    '''

    def __init__(self, environments=[]):
        '''
        '''

        self.environments = environments

    def get_environment(self):
        '''
        Method to obtain the lab manager's env space.
        '''

        # obtain the number of environments available
        n_envs = len(self.environments)

        # the env space for the lab manager consists of 1 number:
        #   env_space[0] is the index of the environment to access

        env_space = gym.spaces.MultiDiscrete([n_envs])

        return env_space

    def get_action_spaces(self, env_index=0):
        '''
        '''

        # obtain the action spaces for each environment
        action_spaces = []
        for env in self.environments:
            action_space = env.action_space
            action_spaces.append(action_space)

        return action_spaces

    def get_observation_space(self):
        '''
        '''

    def reset(self):
        '''
        '''

    def perform_action(self):
        '''
        '''

    def done_reward(self):
        '''
        '''