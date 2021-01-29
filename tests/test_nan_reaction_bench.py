import gym
import sys
sys.path.append('../')
sys.path.append('../chemistrylab/reactions')
import chemistrylab
import numpy as np

env = gym.make('WurtzReact_overlap-v0')
render_mode = "human"
for episode in range(100):
    state = env.reset()
    done = False
    print(episode)
    step = 0
    while not done:
        action = env.action_space.sample()
        state_1, reward, done, _ = env.step(action)
        for s in state_1:
            if np.isnan(s):
                print(step)
                print('===================================')
                print(state)
                print('----------')
                print(action)
                print('----------')
                print(state_1)
                print('===================================')
                assert False
        if done:
            print("done")
        else:
            step += 1
            state = state_1