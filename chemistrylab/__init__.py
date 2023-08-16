'''
ChemistryLab Init Registration File

:title: __init__.py

:author: Chris Beeler and Mitchell Shahen

:history: 2020-07-03
'''

from gymnasium.envs.registration import register

############################ ExtractBench ####################################

register(
    id='GenWurtzExtract-v2',
    entry_point='chemistrylab.benches.extract_bench:GeneralWurtzExtract_v2',
)

register(
    id='WaterOilExtract-v0',
    entry_point='chemistrylab.benches.extract_bench:WaterOilExtract_v0',
)

############################ ReactBench ####################################

register(
    id='GenWurtzReact-v2',
    entry_point='chemistrylab.benches.reaction_bench:GeneralWurtzReact_v2'
)

register(
    id='GenWurtzReact-v0',
    entry_point='chemistrylab.benches.reaction_bench:GeneralWurtzReact_v0'
)

register(
    id='FictReact-v2',
    entry_point='chemistrylab.benches.reaction_bench:FictReact_v2'
)


############################ DistillationBench ####################################

register(
    id='GenWurtzDistill-v2',
    entry_point='chemistrylab.benches.distillation_bench:GeneralWurtzDistill_v2',
    max_episode_steps=100
)

############################ LabManager ####################################

#register(
#    id='LabManager-v0',
#    entry_point='chemistrylab.manager.manager:LabManager',
#    max_episode_steps=100
#)


######################## Bandit Gyms ############################################

register(
    id='FictReactBandit-v0',
    entry_point='chemistrylab.benches.reaction_bench:FictReactBandit_v0'
)

register(
    id='FictReactBandit-v1',
    entry_point='chemistrylab.benches.reaction_bench:FictReactBandit_v0',
    kwargs = dict(targets=["fict_I"])
)

######################## DEMO GYMS #########################################


register(
    id='WurtzExtractDemo-v0',
    entry_point='chemistrylab.benches.extract_bench:WurtzExtractDemo_v0',
)


register(
    id='WurtzReactDemo-v0',
    entry_point='chemistrylab.benches.reaction_bench:WurtzReactDemo_v0',

)

register(
    id='FictReactDemo-v0',
    entry_point='chemistrylab.benches.reaction_bench:FictReactDemo_v0',

)

register(
    id='WurtzDistillDemo-v0',
    entry_point='chemistrylab.benches.distillation_bench:WurtzDistillDemo_v0',
)


register(
    id='ExtractTest-v0',
    entry_point='chemistrylab.benches.extract_bench:SeparateTest_v0',
)


############################# LAB MANAGER #########################################
import os
register(
    id='Manager-v0',
    entry_point='chemistrylab.lab.manager:Manager',
    #this is bad cuz the same benches will be shared across all manager instances
    kwargs = dict(
        config_filename = os.path.dirname(__file__)+"/lab/manager_configs/wurtz.json"
    )
)