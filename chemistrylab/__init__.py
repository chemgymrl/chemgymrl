from gym.envs.registration import register

############################ ExtractWorld ####################################

register(
    id='WurtzExtract-v1',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:ExtractWorld_v1',
    max_episode_steps=100
)

register(
    id='Oil_Water_Extract-v1',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:ExtractWorld_v2',
    max_episode_steps=100
)

############################ ODEWorld ####################################

register(
    id='WurtzReact-v0',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v0:ReactionBenchEnv_0',
    max_episode_steps=20
)

register(
    id='WurtzReact_overlap-v0',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v0:ReactionBenchEnv_0_Overlap',
    max_episode_steps=20
)