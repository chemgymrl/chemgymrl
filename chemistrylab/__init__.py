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
    entry_point='chemistrylab.extract_bench.extract_bench_v1:WurtzExtract_v1',
    max_episode_steps=100
)

register(
    id='WurtzExtract_Ctd-v1',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:ExtractWorld_Wurtz_Ctd_v1',
    max_episode_steps=100
)

register(
    id='Oil_Water_Extract-v1',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:ExtractWorld_Oil_v1',
    max_episode_steps=100
)

register(
    id='MethylRed_Extract-v1',
    entry_point='chemistrylab.extract_bench.methyl_red:ExtractWorld_MethylRed',
    max_episode_steps=100
)

register(
    id='MethylRed_Extract-v2',
    entry_point='chemistrylab.extract_bench.extraction_0:ExtractWorld_MethylRed',
    max_episode_steps=100
)

############################ ReactBench ####################################

register(
    id='WurtzReact-v1',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v1:WurtzReact_v1',
    max_episode_steps=100
)

register(
    id='DecompReact-v0',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v1:ReactionBenchEnv_1',
    max_episode_steps=100
)

register(
    id='WurtzReact-v2',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v1:ReactionBenchEnv_3',
    max_episode_steps=100
)

register(
    id='ODETestReact-v0',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v1:ReactionBenchEnv_ODE_Test',
    max_episode_steps=100
)

############################ DistillationBench ####################################

register(
    id='WurtzDistill-v1',
    entry_point='chemistrylab.distillation_bench.distillation_bench_v1:WurtzDistill_v1',
    max_episode_steps=100
)

############################ LabManager ####################################

register(
    id='LabManager-v0',
    entry_point='chemistrylab.manager.manager_v1:LabManager',
    max_episode_steps=100
)
