import numpy as np
import gym
import gym.spaces
from chemistrylab.extract_bench.extract_bench_v1_engine import ExtractBenchEnv
from chemistrylab.chem_algorithms import material, util, vessel

# initialize extraction vessel
extraction_vessel = vessel.Vessel(label='extraction_vessel',
                                  )

# initialize materials
H2O = material.H2O
HCl = material.HCl
H = material.H
Cl = material.Cl
MethylRed = material.MethylRed
EthylAcetate = material.EthylAcetate

# material_dict
material_dict = {H2O().get_name(): [H2O, 27.7],
                 H().get_name(): [H, 2.5e-4],
                 Cl().get_name(): [Cl, 2.5e-4],
                 MethylRed().get_name(): [MethylRed, 9.28e-4],
                 }
solute_dict = {H().get_name(): {H2O().get_name(): [27.7, 'mol']},
               Cl().get_name(): {H2O().get_name(): [27.7, 'mol']},
               MethylRed().get_name(): {H2O().get_name(): [500, 'ml']},
               }

material_dict, solute_dict, _ = util.check_overflow(material_dict=material_dict,
                                                    solute_dict=solute_dict,
                                                    v_max=extraction_vessel.get_max_volume(),
                                                    )

event_1 = ['update material dict', material_dict]
event_2 = ['update solute dict', solute_dict]
event_3 = ['fully mix']

extraction_vessel.push_event_to_queue(events=None, feedback=[event_1], dt=0)
extraction_vessel.push_event_to_queue(events=None, feedback=[event_2], dt=0)
extraction_vessel.push_event_to_queue(events=None, feedback=[event_3], dt=0)


class ExtractWorld_MethylRed(ExtractBenchEnv):
    def __init__(self):
        super(ExtractWorld_MethylRed, self).__init__(
            extraction='extraction_0',
            extraction_vessel=extraction_vessel,
            target_material='methyl red',
            extractor=EthylAcetate
        )
