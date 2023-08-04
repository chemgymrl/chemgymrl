[chemgymrl.com](https://chemgymrl.com/)

# Distillation Bench

<span style="display:block;text-align:center">![Distillation](tutorial_figures/distillation.png)

The distillation bench provides another set of tools aimed at isolating a requested desired material. The distillation bench utilizes the differing boiling points of materials in the boiling vessel (first slot) to separate materials between vessels.
 
A simple distillation experiment is boiling salt water thus evaporating the water into the air, or into a secondary vessel, leaving only the salt in the initial container. Similarly, the distillation bench obtains a vessel and gradually increases the vessel’s temperature incrementally boiling off materials one at a time. The materials, in their gaseous form, are deposited into an auxiliary vessel, which can be dumped into another auxiliary vessel for storage or removal.

An agent tasked to operate on this bench must control the heat energy added to the vessel as well as the movement of materials between the auxiliary vessels to isolate the desired material with as much purity as possible and spread about as few vessels as possible. Also required of the agent is monitoring the costs associated with adding heat energy and maintenance of unwanted materials. Positive and negative outcomes are associated with actions and operations that lead to the desired material being isolated and of high purity, and thoroughly mixed with unwanted materials and spread about several vessels, respectively.

## Input 

The construction of the distillation bench is initialized in the [distillation_bench](GeneralWurtzDistill_v2) file.

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

Here we include the boiling vessel, as well as two empty vessels. We also provide a precipitation reaction to model how various materials dissolve and precipitate out of the solvents in each vessel. Like in the other benches, we also pass the target material. Additionally, include a heating action for the boiling vessel, and actions to transfer contents between vessels. Heating is simulated by assuming the vessel is in contact with a reservoir and applying Newton's law of heat transfer (as well as enthalpy of vaporization & heat capacity rules).


## Output

Once the distillation bench is reset and the render function is called, plots will appear showing data about the distillation 
being performed by the agent.

- Render
    -  Plots the solvent contents of each vessel, some thermodynamic information, the amount of each material in each vessel.
    The full render plots a significant amount of data for a more in-depth understanding of the information portrayed.

![full render output](tutorial_figures/distillation/full_render_distillation.png)


## Reward
For the reaction benches the default reward function is:

[RewardGenerator](RewardGenerator)(use_purity=True,exclude_solvents=False,include_dissolved=True)

Here, the goal is to maximize the amount and purity of the desired material. In this case, having solvents in the same vessel as the desired material (as well as any other material) will reduce purity.
