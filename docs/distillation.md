[chemgymrl.com](https://chemgymrl.com/)

# Distillation Bench

<span style="display:block;text-align:center">![Distillation](tutorial_figures/distillation.png)

The distillation bench provides another set of experimentation aimed at isolating a requested desired material. Similar to the reaction and extraction benches, a vessel containing materials including the desired material is required as input into this bench. The distillation bench utilizes the differing boiling points of materials in the inputted vessel to separate materials between vessels. The intended output from the distillation bench is a vessel containing a sufficiently high purity and amount of the requested desired material.
 
A simple distillation experiment is boiling salt water thus evaporating the water into the air, or into a secondary vessel, leaving only the salt in the initial container. Similarly, the distillation bench obtains a vessel and gradually increases the vessel’s temperature incrementally boiling off materials one at a time. The materials, in their gaseous form, are deposited into an auxiliary vessel, which can be dumped into another auxiliary vessel for storage or removal.

An agent tasked to operate on this bench must control the heat energy added to the vessel as well as the movement of materials between the auxiliary vessels to isolate the desired material with as much purity as possible and spread about as few vessels as possible. Also required of the agent is monitoring the costs associated with adding heat energy and maintenance of unwanted materials. Positive and negative outcomes are associated with actions and operations that lead to the desired material being isolated and of high purity, and thoroughly mixed with unwanted materials and spread about several vessels, respectively.

## Input 

The construction of the distillation bench is initialized in the `distillation_bench_v1.py` file.

```python
class WurtzDistillDemo_v0(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 60,
    }

    def __init__(self):
        d_rew= RewardGenerator(use_purity=True,exclude_solvents=False,include_dissolved=True)
        shelf = VariableShelf( [
            lambda x:wurtz_vessel(x)[0],
            lambda x:vessel.Vessel("Beaker 1"),
            lambda x:vessel.Vessel("Beaker 2"),
        ],[], n_working = 3)

        amounts=np.ones([1,1])*0.02
        
        heat_info = np.array([
            [300,20], # Room temp water
            [270,20], # Freezing water
            [1000,3]  # Fire
        ])

        actions = [
            Action([0],    heat_info,            'heat contact',   [1],   0.01,   False),
            Action([0],    amounts,              'pour by volume', [1],   0.01,   False),
            Action([1],    amounts,              'pour by volume', [2],   0.01,   False),
            Action([0],    [[0]],                'mix',            None,  0,      True)
        ]
        
        targets = ["dodecane", "5-methylundecane", "4-ethyldecane",
            "5,6-dimethyldecane", "4-ethyl-5-methylnonane", "4,5-diethyloctane", "NaCl"]

        react_info = ReactInfo.from_json(REACTION_PATH+"/precipitation.json")
        reaction = Reaction(react_info)
        reaction.solver="newton"
        reaction.newton_steps=100

        super(WurtzDistillDemo_v0, self).__init__(
            shelf,
            actions,
            ["layers","PVT","targets"],
            targets=targets,
            default_events = (Event("react", (reaction,), None),),
            reward_function=d_rew,
            max_steps=500
        )

    def get_keys_to_action(self):
        # Control with the numpad or number keys.
        keys = {(ord(k),):i for i,k in enumerate("123456") }
        keys[()]=0
        return keys
```

Here we pass the boiling vessel, or a path to the pickle file produced by a previous bench. We provide a reaction
file which identifies the possible targets we are interested in. We also provide a precipitation file which is a
reaction file specifically for describing how various materials dissolve and precipitate out of solution. Like in
the other benches, we also pass the target material. Additionally, we pass in a dQ value which is the maximal change
in heat energy and the path which the output vessel will be located in.

### Distillation Process Explained

In the distillation environment there are 3 main containers or vessels.

- boiling vessel (BV)
- beaker 1       (B1)
- beaker 2       (B2)

The boiling vessel (BV) contains all the materials at the initial state of the experiment. Beaker 1 (B1) can be thought of as a  condensation vessel which is connected to the distillation vessel via a tube and this will contain all the materials that are being boiled off. Beaker 2 (B2) is then the storage vessel, where the condensation vessel can be emptied, in order to make room for other material.

![vessels](https://cdn.pixabay.com/photo/2013/07/13/13/59/chemistry-161903__340.png)

<a style="font-size: 10px">(source: https://pixabay.com/vectors/chemistry-mixture-bulb-violet-161903/)</a>

The point of the process is to extract a target material from the boiling vessel, which contains numerous materials, and we do this by utilizing the different material's boiling points. Typically the process begins by raising the temperature of the BV which allows certain materials in that vessel to boil off into the condensation vessel or B1.

![boiling vessel](https://cdn.pixabay.com/photo/2017/12/27/10/57/chemical-3042414_960_720.png)

<a style="font-size: 10px">(source: https://pixabay.com/illustrations/chemical-equipment-chemistry-glass-3042414/)</a>

As a material's boiling point is reached, any more temperature added from this point will act to evaporate it. The now gaseous material will rise out of the boiling vessel into the tube that feeds into the condensation vessel where it will condense back into its liquid form. In this virtual experiment  it is assumed that this takes place instantaneously. The amount of material evaporated is dependent on the enthalpy of vapour of material being evaporated.

![distillation process](https://cdn.pixabay.com/photo/2013/07/13/13/48/chemistry-161575_960_720.png)

<a style="font-size: 10px">(source: https://pixabay.com/vectors/chemistry-distillation-experiment-161575/)</a>

Once the entirety of the material has been boiled off, the condensation vessel is drained into the storage vessel. Now
the condensation vessel is empty, the boiling vessel's temperature can then be raised more until the next lowest boiling point is reached, thus repeating the process.

![evaporation](https://static.thenounproject.com/png/1639425-200.png)

<a style="font-size: 10px">(source: https://thenounproject.com/term/water-evaporate/1639425/.)</a>

The process is repeated until the desired material has been completely evaporated from the boiling vessel into  condensation vessel. From this point on the desired material is completely isolated and we obtain a hopefully pure sample. We can then choose to end the experiment.

In [lesson 3](https://chemgymrl.readthedocs.io/en/latest/lesson_3_distillation/) in these sets of tutorial for the distillation bench, we will try to get a high reward by obtaining a high molar amount of pure dodecane in our condensation vessel.

For this tutorial, we will just familiarize ourselves with the basic actions, fundamental theory behind distillation, and how you can run the environment on your own!

Here we have the different possible actions that we can take with the environment. The **action_set is an array indexed correspondingly to the action we want to perform.**

The action_space size is equal to the total amount of action parameter tuples.


For example, the following code defines 3 actions: 

```python
heat_info = np.array([
    [300,20], # Room temp water
    [270,20], # Freezing water
    [1000,3]  # Fire
])

Action([0], heat_info, 'heat contact', [1], 0.01, False)

```

The first action (with parameters [300, 20]) corresponds to performing heat transfer with a reservoir at 300 Kelvin (just above room temperature) for 20 time-like units.

Typically an agent will choose actions based on what will give a higher reward, and higher reward is given by getting a high molar amount and concentration of the desired material (in our case dodecane) in a particular vessel.

## Output

Once the distillation bench is reset and the render function is called, plots will appear showing data about the distillation 
being performed by the agent.

- Render
    -  Plots the solvent contents of each vessel, some thermodynamic information, the amount of each material in each vessel.
    The full render plots a significant amount of data for a more in-depth understanding of the information portrayed.

![full render output](tutorial_figures/distillation/full_render_distillation.png)

