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

register(
    id='ODEWorld_1-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_1',
    max_episode_steps=20,
)

register(
    id='ODEWorld_1_overlap-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_1_Overlap',
    max_episode_steps=20,
)

register(
    id='ODEWorld_2-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_2',
    max_episode_steps=20,
)

register(
    id='ODEWorld_2_overlap-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_2_Overlap',
    max_episode_steps=20,
)

register(
    id='ODEWorld_3-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_3',
    max_episode_steps=20,
)

register(
    id='ODEWorld_3_overlap-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_3_Overlap',
    max_episode_steps=20,
)

register(
    id='ODEWorld_4-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_4',
    max_episode_steps=20,
)

register(
    id='ODEWorld_4_overlap-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_4_Overlap',
    max_episode_steps=20,
)

register(
    id='ODEWorld_5-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_5',
    max_episode_steps=20,
)

register(
    id='ODEWorld_5_overlap-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_5_Overlap',
    max_episode_steps=20,
)

register(
    id='ODEWorld_6-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_6',
    max_episode_steps=20,
)

register(
    id='ODEWorld_6_overlap-v0',
    entry_point='chemistrylab.odeworldgym.odeworld_v0:ODEWorldEnv_6_Overlap',
    max_episode_steps=20,
)
