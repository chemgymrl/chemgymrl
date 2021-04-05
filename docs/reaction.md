[chemgymrl.com](https://chemgymrl.com/)

## Reaction Bench

The reaction bench is intended to receive an input vessel container and perform a series of reactions on the materials in that vessel, step-by-step, in the aim to yield a sufficient amount of a desired material. After performing sufficiently many reactions to acquire the desired material, the reaction bench outputs a vessel containing materials, including reactants and products in some measure, that may be operated on in subsequent benches.

The inputted vessel is interpreted to be frozen in time and has been allowed to evolve or react in any way. Once the vessel is subjected to the reaction bench, time is allowed to flow and reactions are allowed to occur resulting in changes in the amounts of materials in the vessel. Once the agent operating in the reaction bench deems that a sufficient number of reaction iterations have occurred, the vessel is once again frozen in time, removed from the reaction bench, and put into storage ready to be withdrawn and inputted into other benches.

Put simply, the reaction bench allows reactions to occur and products to be generated. The reactions that are to occur in the reaction bench are determined by selecting a family of reactions from a directory of supported, available reactions. Moreover, if no materials in the input vessel are involved in the selected family of reactions, no reactions will occur. The agent operating on this bench has the ability to modify reaction parameters, such as the volume and temperature of the vessel to promote or demote reactions thus increasing or decreasing the yield of certain reactants and products. The key to the agent’s success in this bench is to learn how best to allow certain reactions to occur such that the yield of the desired product at the conclusion of the experiment is maximized.

When the agent performs enough positive actions and is satisfied with the amount of the desired result, it concludes the experiment and outputs the vessel. However, the reward system is not as simple as maximizing the amount of a material. Also involved are the costs associated with running the experiment and leveraging the costs of the input materials provided in the initial vessel. Managing these costs and benefits is part of the agent learning how best to operate in the reaction bench.

## Input

The input to the reaction bench is initialized in the `reaction_bench_v1.py` file. 

![reaction bench input](../tutorial_figures/reaction_bench_input.png)

In here we pass parameters such as the materials and solutes needed for the experiment, the path to an input vessel 
(if we're including one), the output vessel path, the number of time steps to be taken during each action, the amount
of time taken in each time step, and an indication to show if spectral plots show overlapping. 

We also pass in the reaction file identifier which is the name of the reaction file located in the 
`../available_reactions` directory. This file includes important values that the engine will use to simulate the 
reaction. It contains information on the reactants, products, and solutes available to the reaction. It defines the 
desired material which an agent will be rewarded for if it picks the actions that produces the said material. It also 
contains all the thermodynamic values and vessel parameters. Lastly it includes arrays for the activation energy, 
stoichiometric coefficients, and concentration calculation coefficients. 

## Output

Once the reaction bench is ran and the render function is called, plots will appear showing data about the reaction 
being performed by the agent. There are two main plot modes:

- Human Render
    - Plots thermodynamic variables and spectral data. The human render plots a minimal amount of data and provides a 
    'surface-level' understanding of the information portrayed.
    - Plots absorbance, time, temperature, volume, pressure, and the amount of reactants remaining.
  
![human render output](../tutorial_figures/human_render_reaction.png)

- Full Render
    -  Plots thermodynamic variables and spectral data. The full render plots a significant amount of data for a more 
    in-depth understanding of the information portrayed.
    - In addition to the data human render plots, full render also plots the molar amount, molar concentration, and 
    absorbance of both reactants and products. It also plots the temperature and pressure mapped to range between 0 and 
    1, as well as the pressure in units kPa.            

![full render output](../tutorial_figures/full_render_reaction.png)
