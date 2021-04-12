[chemgymrl.com](https://chemgymrl.com/)

## Characterization Bench

<span style="display:block;text-align:center">![Characterization](tutorial_figures/characterization.png)

The characterization bench is the primary method in which an agent or lab manager can look inside vessel containers. The characterization bench does not manipulate the inputted vessel in any way, yet subjects it to analysis techniques that observe the state of the vessel including the materials inside it and their relative quantities. This allows an agent or lab manager to observe vessels, determine their contents, and allocate the vessel to the necessary bench for further experimentation.
 

Included with the characterization bench are several analysis techniques available to the agent working at this bench. The foremost of which is spectrometric analysis. Such analysis includes the observation of spectra being emitted by materials. Performing spectrometry techniques on a vessel produces spectral signatures giving insight into the contents of the vessel. Different materials possess different spectral signatures and cross-checking the produced spectral data with the known spectral signatures of supported materials allows the agent to identify the vessel’s contents. Additional characterization bench techniques are in development.
 

In observing the state of the vessel, the agent or lab manager operating this bench can allocate the vessel to another bench for further experimentation in the pursuit of completing the required task. For example, in the extraction of salt from water, if a vessel is determined to have much more water than salt, the lab manager can use this information to perform further extractions, increasing the purity of salt in the vessel, or subject the vessel to the reaction bench and increase the amount of salt present in the vessel.

## Input

The input to the characterization bench is a vessel, and a desired analysis technique to perform on the vessel. These are 
passed in as parameters the analyze method is called.

![analyze method](../tutorial_figures/characterization/analyze.png)

## Output

The output to the characterization bench will depend on the analysis technique called. For example, if the 
characterization bench is called to perform an absorption spectra analysis, it will return a spectral graph back to the 
agent. This output is an array filled spectral signatures.

![absorb](../tutorial_figures/characterization/get_spectra.png)

Note the figure above only shows a snippet of the values returned, in reality the array is much larger.