'''
ChemistryLab Init Registration File

:title: __init__.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-07-03
'''

from gym.envs.registration import register

############################ ExtractBench ####################################

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

############################ ReactBench ####################################

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

############################ DistillationBench ####################################

register(
    id='Distillation-v0',
    entry_point='chemistrylab.distillation_bench.distillation_bench_v1:Distillation_v1',
    max_episode_steps=20
)

############################ LabManager ####################################

register(
    id='LabManager-v0',
    entry_point='chemistrylab.manager.manager_v1:LabManager',
    max_episode_steps=20
)
