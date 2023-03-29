'''
ChemistryLab Init Registration File

:title: __init__.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-07-03
'''

from gym.envs.registration import register

############################ ExtractBench ####################################

register(
    id='GenWurtzExtract-v1',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:GeneralWurtzExtract_v1',
)

register(
    id='WurtzExtract-v1',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:WurtzExtract_v1',
)


register(
    id='GenWurtzExtract-v2',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:GeneralWurtzExtract_v2',
)

register(
    id='WurtzExtract-v2',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:WurtzExtract_v2',
)

register(
    id='WurtzExtractCtd-v1',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:WurtzExtractCtd_v1',
    max_episode_steps=100
)

register(
    id='WaterOilExtract-v1',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:WaterOilExtract_v1',
    max_episode_steps=100
)

############################ ReactBench ####################################

register(
    id='WurtzReact-v1',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v1:WurtzReact_v1',
    max_episode_steps=100
)

register(
    id='GenWurtzReact-v1',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v1:GeneralWurtzReact_v1'
)

register(
    id='FictReact-v1',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v1:FictReact_v1'
)

register(
    id='FictReact-v2',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v1:FictReact_v2'
)

register(
    id='DecompReact-v0',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v1:ReactionBenchEnv_1',
    max_episode_steps=100
)

register(
    id='WurtzReact-v2',
    entry_point='chemistrylab.reaction_bench.reaction_bench_v1:WurtzReact_v2',
    max_episode_steps=100
)

############################ DistillationBench ####################################

register(
    id='WurtzDistill-v1',
    entry_point='chemistrylab.distillation_bench.distillation_bench_v1:WurtzDistill_v1',
    max_episode_steps=100
)

register(
    id='GenWurtzDistill-v1',
    entry_point='chemistrylab.distillation_bench.distillation_bench_v1:GeneralWurtzDistill_v1',
    max_episode_steps=100
)

register(
    id='Distillation-v1',
    entry_point='chemistrylab.distillation_bench.distillation_bench_v1:Distillation_v1',
    max_episode_steps=100
)

############################ LabManager ####################################

register(
    id='LabManager-v0',
    entry_point='chemistrylab.manager.manager_v1:LabManager',
    max_episode_steps=100
)

######################## Discretized ######################################

register(
    id='DiscreteGenWurtzExtract-v0',
    entry_point='chemistrylab.make_discrete:DiscreteWrapper',
    kwargs=dict(entry_point='chemistrylab.extract_bench.extract_bench_v1:GeneralWurtzExtract_v1')
)

register(
    id='DiscreteGenWurtzDistill-v0',
    entry_point='chemistrylab.make_discrete:DiscreteWrapper',
    kwargs=dict(entry_point='chemistrylab.distillation_bench.distillation_bench_v1:GeneralWurtzDistill_v1'),
    max_episode_steps=100
)

register(
    id='DiscreteWurtzExtract-v0',
    entry_point='chemistrylab.make_discrete:DiscreteWrapper',
    kwargs=dict(entry_point='chemistrylab.extract_bench.extract_bench_v1:WurtzExtract_v1')
)

#v1 removes some duplicate actions
register(
    id='DiscreteWurtzExtract-v1',
    entry_point='chemistrylab.make_discrete:DiscreteWrapper',
    kwargs=dict(entry_point='chemistrylab.extract_bench.extract_bench_v1:WurtzExtract_v1',exclude=4)
)

register(
    id='DiscreteWurtzDistill-v0',
    entry_point='chemistrylab.make_discrete:DiscreteWrapper',
    kwargs=dict(entry_point='chemistrylab.distillation_bench.distillation_bench_v1:WurtzDistill_v1'),
    max_episode_steps=100
)
#v1 removes some duplicate actions
register(
    id='DiscreteWurtzDistill-v1',
    entry_point='chemistrylab.make_discrete:DiscreteWrapper',
    kwargs=dict(entry_point='chemistrylab.distillation_bench.distillation_bench_v1:WurtzDistill_v1',exclude=9),
    max_episode_steps=100
)

register(
    id='DiscreteGenWurtzReact-v1',
    entry_point='chemistrylab.make_discrete:DiscreteWrapper',
    kwargs=dict(entry_point='chemistrylab.reaction_bench.reaction_bench_v1:GeneralWurtzReact_v1',null_act=(0.5,0.5,0,0,0,0))
)






