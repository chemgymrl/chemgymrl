# ChemGymRL

[![Build Status](https://travis-ci.com/chemgymrl/chemgymrl.svg?branch=main)](https://travis-ci.com/chemgymrl/chemgymrl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chemgymrl.readthedocs.io/en/latest/)

## Overview

Check out our website at [chemgymrl.com](https://www.chemgymrl.com)

ChemGymRL is a collection of reinforcement learning environments in the open AI standard which are designed for experimentation and exploration with reinforcement learning in the context of chemical discovery. These experiments are virtual variants of experiments and processes that would otherwise be performed in real-world chemistry labs and in industry.

ChemGymRL simulates real-world chemistry experiments by allocating agents to benches to perform said experiments as dictated by a lab manager who has operational control over all agents in the environment. The environment supports the training of Reinforcement Learning agents by associating positive and negative rewards based on the procedure and outcomes of actions taken by the agents. For more information, visit our documentation at [https://chemgymrl.readthedocs.io/en/latest/](https://chemgymrl.readthedocs.io/en/latest/).

### Bench Overview

The ChemGymRL environment consists of four benches and a lab manager tasked with organizing the experimentation occurring at each bench. The four benches available to agents under the direction of the lab manager include: the reaction bench, the extraction bench, the distillation bench, and the characterization bench. For more information about these benches, be sure to check out our documentation [here](https://chemgymrl.readthedocs.io/en/latest/).

#### Reaction Bench

The reaction bench is intended to receive an input vessel container and perform a series of reactions on the materials in that vessel, step-by-step, in the aim to yield a sufficient amount of a desired material. After performing sufficiently many reactions to acquire the desired material, the reaction bench outputs a vessel containing materials, including reactants and products in some measure, that may be operated on in subsequent benches.

#### Extraction Bench

The extraction bench aims to isolate and extract certain materials from an inputted vessel container containing multiple materials. This is done by means of transferring materials between a number of vessels and utilizing specifically selected solutes to demarcate and separate materials from each other. The intended output of this bench is at least one beaker, each containing a desired material in quantities that exceed a minimum threshold.

#### Distillation Bench

The distillation bench provides another set of experimentation aimed at isolating a requested desired material. Similar to the reaction and extraction benches, a vessel containing materials including the desired material is required as input into this bench. The distillation bench utilizes the differing boiling points of materials in the inputted vessel to separate materials between vessels. The intended output from the distillation bench is a vessel containing a sufficiently high purity and amount of the requested desired material.

#### Characterization Bench

The characterization bench is the primary method in which an agent or lab manager can look inside vessel containers. The characterization bench does not manipulate the inputted vessel in any way, yet subjects it to analysis techniques that observe the state of the vessel including the materials inside it and their relative quantities. This allows an agent or lab manager to observe vessels, determine their contents, and allocate the vessel to the necessary bench for further experimentation.

#### Lab Manager

The lab manager acts as another agent in the ChemGymRL environment, yet the lab manager’s task is to organize and dictate the activity of bench agents, input and output vessels, benches, and bench environments.

### Reward Function

The reward function is crucial such that it is how the agents determine which actions are seen as positive and which actions are seen as negative. The current hierarchy of positive states has the amount of desired material present be of greater importance compared to the number of vessels containing the desired material. That being said, vessels containing of purity of less than 20%, with respect to the desired material, are considered unfit for continuation. For example, suppose performing actions in the extraction bench yields a vessel with 50% purity and another vessel with 10% purity. The purity threshold is arbitrary, so, for the sake of this example, let's suppose the purity threshold is set at 20%. The vessel with 50% purity exceeds the purity requirement and is exempt from further actions in the extraction bench, however the vessel with 10% purity must still undergo extraction bench actions to increase it's purity to the required 20%.

### Vessel

The Vessel class serves as any container you might find in a lab, a beaker, a dripper, etc. The vessel class simulates and allows for any action that you might want to perform within a lab, such as draining contents, storing gasses from a reaction, performing reactions, mix, pour, etc. For more information about the vessel class, make sure to check out our documentation at [https://chemgymrl.readthedocs.io/en/latest/vessel_lesson/](https://chemgymrl.readthedocs.io/en/latest/vessel_lesson/)

#### The Workflow
  
  1. Instruct the lab manager to acquire a desired material
  2. The lab manager assigns a vessel and an agent to carry out reactions in the reaction bench
  3. The reaction bench agent performs actions on the vessel generating products including the desired material
  4. The agent gives the resulting vessel to the lab manager
  5. The lab manager analyzes the vessel in the characterization bench determining its contents
  6. Verifying that the vessel contains some of the desired material, the lab manager assigns another agent and the vessel to the extraction bench to isolate the material in high quantities
  7. The extraction bench agent performs actions on the vessel moving its contents between many vessels sequentially isolating the desired material in greater quantities
  8. The extraction bench agent gives the resulting vessel(s) to the lab manager
  9. The lab manager analyzes in the characterization bench determining its contents and quantities
  10. The lab manager selects the vessel with the highest quantity of the desired material and assigns it and an agent to the distillation bench to extract the desired material
  11. The distillation bench agent performs operations on the vessel to heat and boil off materials from the vessel into ancillary vessels
  12. The distillation bench agent passes the resulting vessel to the lab manager
  13. The lab manager subjects the vessel to the characterization bench once again determining its contents
  14. If the lab manager is satisfied that the desired material has been sufficiently isolated, the lab manager concludes the experiment

### Supported Materials

All the materials supported by the environments will be located in `../chemistrylab/chem_algorithms/material.py`. This file contains the material class where materials and their properties can be defined. There are numerous, already defined materials, available for use. These materials contains properties including the name, density, polarity, boiling point, melting point, specific heat, enthalpy of vapor, enthalpy of fusion, and many more. The material properties are listed to be consistent and the intended units are included in the file's docstring.

If you need to add any materials, it is quite easy to do so as most properties are relative straightforward to add. It's important to note that when adding spectra you will need to add the desired spectrum to the `../chemistrylab/ode_algorithms/spectra/diff_spectra.py` file so that the material class can access it from there.

### Expected Output

At the end of an experiment, the user can expect to receive a vessel containing the requested desired material. Such a vessel is stored locally as a pickle file. Additionally, a detailed log message will appear including the final cumulative reward supplied to the lab manager, the purity of the desired material in the output vessel and plots displaying the contents of the vessel and the vessel spectra. Since the outputted vessel is made available to the user, they have the option of performing their own analysis and operations on the vessel with the ChemGymRL environment or subject the vessel to another experiment. For more information about the input of every individual bench, you can check out the documentation at [https://chemgymrl.readthedocs.io/en/latest/](https://chemgymrl.readthedocs.io/en/latest/).


### Installation

#### Clone Repository:

The first step of the install process is to clone the repository, that can be done using the following command line
instructions:
```commandline
cd path/to/desired/install/location
git clone https://github.com/chemgymrl/chemgymrl.git
```
now that the repo has been installed we need to enter into the repository:

```commandline
cd chemgymrl
```

#### Python Virtual Environment:

ChemGymRL is set to use python 3, and more specifically python 3.8. The first step of this next part is to install
[python](https://python.org), if you already have python then the next step is to create a virtual environment using
your favourite virtual environment tool. In this tutorial we will use virtualenv, but anaconda works as well. The next
steps will show you how to create and activate the correct virtual environment:

```commandline
python3.8 -m venv chemgymrl
source chemgymrl/bin/activate
```

Now that the virtual environment is created and activated we will now look to install all the correct packages.

#### Install Library:
Now that everything is set up we simply need to install the library. If you wish to make a lot of changes to the library and implement custom reactions, extractions etc. we recommend that you simply work out of the repository rather than install it as a library.
 
```commandline
pip install .
```
