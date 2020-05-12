import numpy as np
import gym
import gym.spaces
from chemistrylab.extractworldgym.extractworld_v1_engine import ExtractWorldEnv
from chemistrylab.chem_algorithms import material, util, vessel

# initialize extraction vessel
extraction_vessel = vessel.Vessel(label='extraction_vessel',
                                  )

# initialize materials
H2O = material.H2O()
Na = material.Na()
Cl = material.Cl()
Na.set_charge(1.0)
Na.set_solute_flag(True)
Na.set_polarity(2.0)
Cl.set_charge(-1.0)
Cl.set_solute_flag(True)
Cl.set_polarity(2.0)

# material_dict
material_dict = {H2O.get_name(): [H2O, 30.0],
                 Na.get_name(): [Na, 1.0],
                 Cl.get_name(): [Cl, 1.0]
                 }
solute_dict = {Na.get_name(): {H2O.get_name(): 1.0},
               Cl.get_name(): {H2O.get_name(): 1.0},
               }

material_dict, solute_dict, _ = util.check_overflow(material_dict=material_dict,
                                                    solute_dict=solute_dict,
                                                    v_max=extraction_vessel.get_max_volume(),
                                                    )

event_1 = ['update material dict', material_dict]
event_2 = ['update solute dict', solute_dict]
event_3 = ['fully mix']

extraction_vessel.push_event_to_queue(events=None, feedback=[event_1, event_2, event_3], dt=0)


class ExtractWorld_v1(ExtractWorldEnv):
    def __init__(self):
        super(ExtractWorld_v1, self).__init__(
            extraction='water_oil_2',
            extraction_vessel=extraction_vessel,
            target_material='Na',
        )
