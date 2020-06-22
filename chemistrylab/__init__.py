from gym.envs.registration import register

############################ ExtractWorld ####################################

register(
    id='ExtractWorld_0-v1',
    entry_point='chemistrylab.extractworldgym.extractworld_v1:ExtractWorld_v1',
    max_episode_steps=100,
)

############################ ODEWorld ####################################

register(
    id='ODEWorld_0-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_0',
    max_episode_steps=20,
)

register(
    id='ODEWorld_0_overlap-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_0_Overlap',
    max_episode_steps=20,
)
