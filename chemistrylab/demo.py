import os
import numpy as np
import gymnasium as gym
import chemistrylab
from chemistrylab.util import Visualization,ActionDoc

if __name__ == "__main__":
    from gymnasium.utils.play import play
    #Visualization.set_backend("numba")
    Visualization.set_backend("pygame")

    Visualization.matplotVisualizer.legend_update_delay=1

    #Visualization.use_mpl_dark()

    while True:
        print("Enter 0 for Extraction, 1 For Distillation, or 2 for Reaction")
        x = input()
        if x in ["0","1","2","3"]:
            break
    x = int(x)
    env_id = ["WurtzExtractDemo-v0","WurtzDistillDemo-v0","FictReactDemo-v0","ExtractTest-v0"][x]

    env=gym.make(env_id)
    env.metadata["render_modes"] = ["rgb_array"]
    _=env.reset()



    if x==2:
        keys = dict()
        for i in range(5):
            arr=np.zeros(5)
            arr[i]=1    
            keys[str(i+1)] = arr
    else:
        keys = {k:i for i,k in enumerate(["1234567890","123456","0","12345qwert890"][x])}


    noop = [7,0,np.zeros(5),10][x]

    print("Actions: (Use the number keys)")

    table = ActionDoc.generate_table(env.shelf,env.actions)
    print(table)

    input("Press Enter to Start")

    ret=0
    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        global ret
        ret+=rew
        if terminated:
            os.system('cls' if os.name=='nt' else 'clear')
            print(ret)
            ret=0
        return [rew]
        
    play(env,keys_to_action=keys,noop=noop,fps=60,callback=callback)