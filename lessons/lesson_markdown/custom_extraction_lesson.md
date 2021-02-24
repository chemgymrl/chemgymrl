## Building A Custom Extraction File

For this tutorial we will be showing you how to create a custom extraction environment to train an RL agent on!

In this case we will be doing the extraction of Methyl Red from a solution of hydrochloric acid into Ethyl Acetate.
The procedure and set up of this reaction can be found at this
[link](https://chem.libretexts.org/Bookshelves/Organic_Chemistry/Book%3A_Organic_Chemistry_Lab_Techniques_(Nichols)/04%3A_Extraction/4.06%3A_Step-by-Step_Procedures_For_Extractions).
Now let's get started.

![extraction environment]()

The extraction bench is one of the more complicated environments to build since there are a lot more actions to take in
the environment and as a result a lot more to set up. As a result we are starting with a very simple, single extraction
but after having implemented this it should be obvious how to extend this to larger more complicated extractions.


Typically, in a new reaction or extraction it will be important to define the new materials in 
`chemistrylab/chem_algorithms/material.py` but for this case we have implemented the 2 new materials
(methyl red and ethyl acetate). Go into the material file and look at the materials and familiarize yourself with the
properties of each material. In your own project after creating the new materials it is then time to create the
extraction environment space, and the actions that can be performed in that extraction. For a simple single extraction
we can use a prebuilt environment in this case `chemistrylab/extract_bench/extraction_0.py`. Please take a read through
the file and try to understand how actions are performed and how we set up vessels. Now we will create a new extraction
bench file in the following directory `chemistrylab/extract_bench`, let's create a new file called `methyl_red.py`.
In this new file we are going to use the following code:

```python
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
# The material dict is the data-structure we use to describe the materials that will be in our vessel
# found inside of our vessel, in this case it is of the form
# {material_name: [material_class, quantity, unit]}
# you don't need to specify a unit but if you don't we assume you are using mols
material_dict = {H2O().get_name(): [H2O, 27.7],
                 H().get_name(): [H, 2.5e-4],
                 Cl().get_name(): [Cl, 2.5e-4],
                 MethylRed().get_name(): [MethylRed, 9.28e-4],
                 }
# solute_dict
# the solute dict describes how different materials are dissolved in others
# {solute_name: {solvent_name: [quantity, unit]}}
# you don't need to specify a unit but if you don't we assume you are using mols
solute_dict = {H().get_name(): {H2O().get_name(): [27.7, 'mol']},
               Cl().get_name(): {H2O().get_name(): [27.7, 'mol']},
               MethylRed().get_name(): {H2O().get_name(): [500, 'ml']},
               }

# this function checks if we have poured too many materials in our vessel and if we have it returns a vessel
# with the appropriate amount of materials lost
material_dict, solute_dict, _ = util.check_overflow(material_dict=material_dict,
                                                    solute_dict=solute_dict,
                                                    v_max=extraction_vessel.get_max_volume(),
                                                    )

# Here we push events that update the material dictionary, solute dictionary and then mix all of the materials together
event_1 = ['update material dict', material_dict]
event_2 = ['update solute dict', solute_dict]
event_3 = ['fully mix']

extraction_vessel.push_event_to_queue(events=None, feedback=[event_1], dt=0)
extraction_vessel.push_event_to_queue(events=None, feedback=[event_2], dt=0)
extraction_vessel.push_event_to_queue(events=None, feedback=[event_3], dt=0)

# Here is our initialization of the extraction bench environment
# Make sure to give it a unique name and specify the correct extraction bench that we have defined above
# in this case our extractor is EthylAcetate because it is less polar than water, this in turn allows us
# to extract the desired material methyl red
class ExtractWorld_MethylRed(ExtractBenchEnv):
    def __init__(self):
        super(ExtractWorld_MethylRed, self).__init__(
            extraction='extraction_0',
            extraction_vessel=extraction_vessel,
            target_material='methyl red',
            extractor=EthylAcetate
        )

```

Now we just have to add some code to allow the new environment to be recognized by gym. In the following file
`chemistrylab/__init__.py` we will add the following line of code:

```python
register(
    id='MethylRed_Extract-v1',
    entry_point='chemistrylab.extract_bench.methyl_red:ExtractWorld_MethylRed',
    max_episode_steps=100
)
```

Now we're done! To test that these changes have worked, simply run the following code:

```python
from gym import envs

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs if 'Extract' in env_spec.id]
print(env_ids)
```
You should see in the output `MethylRed_Extract-v1`
```
# ['WurtzExtract-v1', 'Oil_Water_Extract-v1', 'MethylRed_Extract-v1', 'MethylRed_Extract-v2']
```