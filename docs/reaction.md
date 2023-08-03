[chemgymrl.com](https://chemgymrl.com/)

# Reaction Bench

<span style="display:block;text-align:center">![Reaction](tutorial_figures/reaction.png)

The sole purpose of the reaction bench is to allow the agent to transform available reactants into various products via a chemical reaction. To this end, the agent is able to perform a series of reactions on the materials in a single reaction vessel, step-by-step, with the aim to yield a sufficient amount of some desired material. After performing sufficiently many reactions to acquire the desired material, the reaction vessel may be removed and operated on in subsequent benches.

Before the reaction bench is started, the reaction vessel is interpreted to be frozen in time and has been allowed to evolve or react in any way. Once the reaction bench is initialized, time is allowed to flow and reactions are allowed to occur resulting in changes in the amounts of materials in the vessel. Once the agent operating in the reaction bench deems that a sufficient number of reaction iterations have occurred, the vessel is once again frozen in time, removed from the reaction bench, and put into storage ready to be withdrawn and inputted into other benches.

Put simply, the reaction bench allows reactions to occur and products to be generated. The reactions that are to occur in the reaction bench are determined by selecting a family of reactions from a directory of supported, available reactions. Moreover, if no materials in the input vessel are involved in the selected family of reactions, no reactions will occur. The agent operating on this bench has the ability to modify reaction parameters, such as the temperature of the vessel to promote or demote reactions thus increasing or decreasing the yield of certain reactants and products. The key to the agent’s success in this bench is to learn how best to allow certain reactions to occur such that the yield of the desired product at the conclusion of the experiment is maximized.

When the agent performs enough positive actions and is satisfied with the amount of the desired result, it concludes the experiment and outputs the vessel. However, the reward system may not be as simple as maximizing the amount of a material. Any costs associated with running the experiment and leveraging materials provided may be included. Managing these costs and benefits is part of the agent learning how best to operate in the reaction bench.

## Input

The input to the reaction bench is initialized in the [reaction_bench_v1.py](_modules/chemistrylab/benches/reaction_bench.html#GeneralWurtzReact_v0) file. 

```python
class GeneralWurtzReact_v2(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }
    def __init__(self):
        r_rew = RewardGenerator(use_purity=False,exclude_solvents=False,include_dissolved=False)
        shelf = Shelf([
            get_mat("diethyl ether",4,"Reaction Vessel"),
            get_mat("1-chlorohexane",1),
            get_mat("2-chlorohexane",1),
            get_mat("3-chlorohexane",1),
            get_mat("Na",3),
        ])
        actions = [
            Action([0],    [ContinuousParam(156,307,0,(500,))],  'heat contact',   [0],  0.01,  False),
            Action([1],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([2],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([3],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
            Action([4],    [ContinuousParam(0,1,1e-3,())],      'pour by percent',  [0],   0.01,   False),
        ]

        react_info = ReactInfo.from_json(REACTION_PATH+"/chloro_wurtz.json")
        
        super(GeneralWurtzReact_v2, self).__init__(
            shelf,
            actions,
            ["PVT","spectra","targets"],
            targets=react_info.PRODUCTS,
            default_events = (Event("react", (Reaction(react_info),), None),),
            reward_function=r_rew,
            discrete=False,
            max_steps=20
        )
```

Here we pass parameters like vessels containing materials and solutes needed for the experiment, the initial conditions of the reaction vessel, the amount of time taken in each time step, what information is passed along as observations, and what reward scheme should be used. 

We also pass in a reaction event which is constructed using a json file located in `chemistrylab/reactions/available_reactions`. This file includes important values that the engine will use to simulate the reaction. It contains information on the reactants, products, and solutes available for the reaction. Additionally, it includes arrays for the activation energy, stoichiometric coefficients, and concentration calculation coefficients. 

## Output

Once the reaction bench is reset and the render function is called, plots will appear showing data about the reaction 
being performed by the agent.

- Render
    - Plots thermodynamic variables and spectral data. The human render plots a minimal amount of data and provides a 
    'surface-level' understanding of the information portrayed.
    - Plots absorbance, time, temperature, volume, pressure, and the amount of reactants remaining.
  
![human render output](tutorial_figures/reaction/human_render_reaction.png)

## Reward
For the reaction benches the default reward function is:

[RewardGenerator](chemistrylab.util.html#chemistrylab.util.reward.RewardGenerator)(use_purity=False,exclude_solvents=False,include_dissolved=False)

Here, the goal is to maximize the amount of the desired material while possibly minimizing an undesired material.
