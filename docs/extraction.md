[chemgymrl.com](https://chemgymrl.com/)

## Extraction Bench

<span style="display:block;text-align:center">![Extraction](tutorial_figures/extraction.png)

The extraction bench aims to isolate and extract certain materials from an inputted vessel container containing multiple materials. This is done by means of transferring materials between a number of vessels and utilizing specifically selected solutes to demarcate and separate materials from each other. The intended output of this bench is at least one beaker, each containing a desired material in quantities that exceed a minimum threshold.
 
A classic extraction experiment is extracting salt from oil using water. Water is added to a beaker containing salt dissolved in oil, settling below the oil. When the water settles, the dissolved salt is pulled from the oil into the water. Similarly, in the extraction bench, solvents are added to act as the oil and water in the previous extraction experiment, separating collections of materials into layers that can be extracted and poured into auxiliary vessels. An agent performing this experiment will be able to use solutes and extract materials with the aim of extracting and isolating the specified desired material.

The agent operating on this bench will experiment using solutes in different scenarios and moving materials between beakers to learn which actions under which circumstances constitute positive and negative results. Positive results resemble actions that increase the purity of the desired material in a vessel or aim to increase the amount of the desired material in a vessel, while negative results include mixing the desired material with a host of unwanted materials. Like in the reaction bench, purity and amount are not the only variables used to facilitate the agent’s learning. Material costs and time constraints are additional factors that the agent must navigate when attempting to maximize the purity and yield of the desired material. Once the costs to continue outweigh the benefits or the required minimum purity threshold is exceeded, all vessels containing a sufficient amount of the desired material are outputted, stored, and made available for further use.

## Input 

The input to the extraction bench is initialized in the `extraction_bench_v1.py` file.

```
class WurtzExtract_v1(ExtractBenchEnv):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self):
        super(WurtzExtract_v1, self).__init__(
            extraction='wurtz',
            extraction_vessel=wurtz_vessel('dodecane'),
            reaction=_Reaction,
            reaction_file_identifier="chloro_wurtz",
            n_steps=50,
            target_material='dodecane',
            solvents=["C6H14", "DiEthylEther"],
            out_vessel_path=os.getcwd()
        )
```

Here we pass the extraction we want to perform, in the figure above, we would perform the wurtz extraction. The input 
vessel is also passed here. It contains the material and solute dictionary of a vessel that's typically outputted by the
reaction bench. We also pass the solute that the extraction will take place in, as well as the target material, and path 
of the output vessel.

## Output

The extraction bench plots data about the extraction being performed by the agent. Unlike reaction bench, in extraction 
there is only the human render mode.

![extraction output](../tutorial_figures/extraction/human_render_extraction.png)

The output plot shows the solvent contents of each vessel, where each pixel corresponds to a solvent. Sequential pixels
corresponding to the same solvent constitute a single layer.

Like reaction bench the extraction bench also outputs a pickle file once the extraction process is completed. The
default name for this file is `extract_vessel_{i}` where i ranges from 0 to the total number of validated vessels.
