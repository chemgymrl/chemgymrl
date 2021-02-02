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
Na = material.Na
Cl = material.Cl
Li = material.Li
F = material.F

# material_dict
material_dict = {H2O().get_name(): [H2O, 30.0],
                 Na().get_name(): [Na, 1.0],
                 Cl().get_name(): [Cl, 1.0],
                 Li().get_name(): [Li, 1.0],
                 F().get_name(): [F, 1.0]
                 }
solute_dict = {Na().get_name(): {H2O().get_name(): [1.0]},
               Cl().get_name(): {H2O().get_name(): [1.0]},
               Li().get_name(): {H2O().get_name(): [1.0]},
               F().get_name(): {H2O().get_name(): [1.0]}
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


class ExtractWorld_Lesson1(ExtractBenchEnv):
    def __init__(self):
        super(ExtractWorld_Lesson1, self).__init__(
            extraction='lesson_1',
            extraction_vessel=extraction_vessel,
            target_material=['Na', 'Li'],
        )
