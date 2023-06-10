[chemgymrl.com](https://chemgymrl.com/)

# Lab / Lab Manager Tutorial

Here is a [link](https://github.com/chemgymrl/chemgymrl/blob/main/lessons/notebooks/Lab%20tutorial.ipynb) to the jupyter notebook, please use it at your pleasure.

The lab environment serves as an environment for a user to conduct chemical experiments as they would in a physical lab, further the lab environment provides the opportunity to train a high-level agent to synthesize materials using the 4 lab benches and the agents associated with each. The environment allows the user to use a variety of reward functions based on the cost of real-world lab equipment, material costs purity, and labour costs. In this tutorial, we will walk through how a lab manager agent would walk through the process of trying to synthesize dodecane. Further, we will walk through the manager wrapper that we have developed which gives a simple api for an agent to run in an environment.

## Lab

First off we can go over the 4 lab benches that the agent will have access to when trying to synthesize a material.

| Bench Index: | Bench Name:            | Bench Description:                                                                                                                                                                                                                                                                                                                                                            |
|--------------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0            | Reaction Bench         | The reaction bench serves as a tool for an agent to operate within an environment where they can learn how to optimize the process by which they synthesize a material. By using a  predefined reaction, we simulate the reaction allowing the agent to learn how best to optimize the reaction.                                                                         |
| 1            | Extraction Bench       | The extraction bench serves as a tool for an agent to learn how to extract materials from one solution to another.                                                                                                                                                                                                                                                          |
| 2            | Distillation Bench     | The distillation bench serves as a tool for an agent to learn how to distill and purify solutions down to a single desired material.                                                                                                                                                                                                                                        |
| 3            | Characterization Bench | The characterization bench is different from the rest of the benches as it doesn't have an agent trained to run in it. In this case, the characterization bench takes in a vessel and the desired analysis technique and returns the results of that analysis, for instance performing an absorption spectra analysis and returning the spectral graph to the agent. |


```python
import chemistrylab
import numpy as np
```


```python
from chemistrylab.lab.lab import Lab
```


```python
lab = Lab()
# Notice how there are currently no vessels in our shelf
lab.shelf.vessels
```

Now that we have initialized the lab environment we should take a look at the action space for the lab environment.

| Bench Index | Bench Env Index | Vessel Index      | Agent Index      |
|-------------|-----------------|-------------------|------------------|
| 0 - 4         | 0 - Max_Num_Env   | 0 - Max_Num_Vessels | 0 - Max_Num_Agents |

So part of the challenge for an agent running in this environment will be that each bench has a different number of bench environments and available agents. For instance, the user could have 10 unique reactions which only require a general extraction and a general distillation, in this case, Max_Num_Env will be 10 even if the extraction bench is selected and even though there is only 1 registered environment for the extraction bench. As such if the user selects an agent or bench environment that is not available for a certain bench the agent will receive a negative reward. Now that we have looked over the action space it will help to look over the lab environment in some examples that will give some more detail.

First, let's take a look at all the environments registered to each bench:


```python
# All reaction environments that are available to the agent
lab.reactions
```

```
['WurtzReact-v1',
 'GenWurtzReact-v1',
 'FictReact-v1',
 'FictReact-v2',
 'DecompReact-v0',
 'WurtzReact-v2']
```


```python
# All extraction environments that are available to the agent
lab.extractions
```

```
['GenWurtzExtract-v1',
 'WurtzExtract-v1',
 'WurtzExtractCtd-v1',
 'WaterOilExtract-v1']
```


```python
# All distillations environments that are available to the agent
lab.distillations
```

```
['WurtzDistill-v1', 'GenWurtzDistill-v1', 'Distillation-v1']
```


Now that we know what environments are registered let's take a look at the available agents:


```python
# All reaction agents
lab.react_agents
```




    {'random': <chemistrylab.lab.agent.RandomAgent at 0x7f4f1de2af40>}




```python
# All extraction agents
lab.extract_agents
```




    {'random': <chemistrylab.lab.agent.RandomAgent at 0x7f4f1de2ab80>}




```python
# All distillation agents
lab.distill_agents
```




    {'random': <chemistrylab.lab.agent.RandomAgent at 0x7f4f1de2ae20>}



Perfect, now that we can understand our action space, let's perform some actions!


```python
react_action = np.array([0, 0, 0, 0])
lab.step(react_action)
```

    WurtzReact-v1





    (1.3435654540975503, array([], dtype=float64), False)



From above we see that the react_action is loading the reaction bench with the WurtzReact-v1 environment, the 0th vessel, and a random agent. From the output we see the following: ((reward, analysis_array), Done). Now that we have run a step, let's take a look at the vessels available to the lab:


```python
lab.shelf.vessels
```


```python
lab.shelf.vessels[0].get_material_dict()
```




    {'1-chlorohexane': [chemistrylab.chem_algorithms.material.OneChlorohexane,
      0.0002588906111498813,
      'mol'],
     '2-chlorohexane': [chemistrylab.chem_algorithms.material.TwoChlorohexane,
      0.00021795409858640458,
      'mol'],
     '3-chlorohexane': [chemistrylab.chem_algorithms.material.ThreeChlorohexane,
      0.0002978148554260234,
      'mol'],
     'Na': [chemistrylab.chem_algorithms.material.Na, 0.0, 'mol'],
     'dodecane': [chemistrylab.chem_algorithms.material.Dodecane,
      0.001834376444740615,
      'mol'],
     '5-methylundecane': [chemistrylab.chem_algorithms.material.FiveMethylundecane,
      0.0015687644112145217,
      'mol'],
     '4-ethyldecane': [chemistrylab.chem_algorithms.material.FourEthyldecane,
      0.0013702501853332663,
      'mol'],
     '5,6-dimethyldecane': [chemistrylab.chem_algorithms.material.FiveSixDimethyldecane,
      0.001419066255007586,
      'mol'],
     '4-ethyl-5-methylnonane': [chemistrylab.chem_algorithms.material.FourEthylFiveMethylnonane,
      0.0012292376915970695,
      'mol'],
     '4,5-diethyloctane': [chemistrylab.chem_algorithms.material.FourFiveDiethyloctane,
      0.0011840664043430795,
      'mol'],
     'NaCl': [chemistrylab.chem_algorithms.material.NaCl,
      0.01721152278447229,
      'mol']}



notice how we now have a vessel in the shelf and when we look at it we can see chemicals from the wurtz reaction. Now that we have these leftover materials in our vessel, we want to try and extract dodecane out of the vessel.


```python
extract_action = np.array([1, 0, 0, 0])
lab.step(extract_action)
```

    WurtzExtract-v1
    (0, array([], dtype=float64), False)



From above we see that the extract_action is loading the extraction bench with the 'WurtzExtract-v1' environment, the 0th vessel (as seen above), and a random agent


```python
lab.shelf.vessels
```


```python
for vessel in lab.shelf.vessels:
    print(vessel.get_material_dict())
    print("_____________")
```

    {'1-chlorohexane': [<class 'chemistrylab.chem_algorithms.material.OneChlorohexane'>, 0.0002588906111498813, 'mol'], '2-chlorohexane': [<class 'chemistrylab.chem_algorithms.material.TwoChlorohexane'>, 0.00021795409858640458, 'mol'], '3-chlorohexane': [<class 'chemistrylab.chem_algorithms.material.ThreeChlorohexane'>, 0.0002978148554260234, 'mol'], 'Na': [<class 'chemistrylab.chem_algorithms.material.Na'>, 0.0, 'mol'], 'dodecane': [<class 'chemistrylab.chem_algorithms.material.Dodecane'>, 0.001834376444740615, 'mol'], '5-methylundecane': [<class 'chemistrylab.chem_algorithms.material.FiveMethylundecane'>, 0.0015687644112145217, 'mol'], '4-ethyldecane': [<class 'chemistrylab.chem_algorithms.material.FourEthyldecane'>, 0.0013702501853332663, 'mol'], '5,6-dimethyldecane': [<class 'chemistrylab.chem_algorithms.material.FiveSixDimethyldecane'>, 0.001419066255007586, 'mol'], '4-ethyl-5-methylnonane': [<class 'chemistrylab.chem_algorithms.material.FourEthylFiveMethylnonane'>, 0.0012292376915970695, 'mol'], '4,5-diethyloctane': [<class 'chemistrylab.chem_algorithms.material.FourFiveDiethyloctane'>, 0.0011840664043430795, 'mol'], 'NaCl': [<class 'chemistrylab.chem_algorithms.material.NaCl'>, 0.01721152278447229, 'mol']}
    _____________
    {}
    _____________
    {}
    _____________


From the above, we can see that 2 new vessels have been added to our shelf courtesy of the extraction bench.


```python
distill_action = np.array([2, 0, 0, 0])
lab.step(distill_action)
```

    Distillation-v0
    (0, array([], dtype=float64), False)



From above we see that the distill_action is loading the distillation bench with the 'Distillation-v0' environment, the 0th vessel (as seen above), and a random agent


```python
for vessel in lab.shelf.vessels:
    print(vessel.get_material_dict())
    print("_____________")
```

    {'1-chlorohexane': [<class 'chemistrylab.chem_algorithms.material.OneChlorohexane'>, 0.0002588906111498813, 'mol'], '2-chlorohexane': [<class 'chemistrylab.chem_algorithms.material.TwoChlorohexane'>, 0.00021795409858640458, 'mol'], '3-chlorohexane': [<class 'chemistrylab.chem_algorithms.material.ThreeChlorohexane'>, 0.0002978148554260234, 'mol'], 'dodecane': [<class 'chemistrylab.chem_algorithms.material.Dodecane'>, 0.001834376444740615, 'mol'], '5-methylundecane': [<class 'chemistrylab.chem_algorithms.material.FiveMethylundecane'>, 0.0015687644112145217, 'mol'], '4-ethyldecane': [<class 'chemistrylab.chem_algorithms.material.FourEthyldecane'>, 0.0013702501853332663, 'mol'], '5,6-dimethyldecane': [<class 'chemistrylab.chem_algorithms.material.FiveSixDimethyldecane'>, 0.001419066255007586, 'mol'], '4-ethyl-5-methylnonane': [<class 'chemistrylab.chem_algorithms.material.FourEthylFiveMethylnonane'>, 0.0012292376915970695, 'mol'], '4,5-diethyloctane': [<class 'chemistrylab.chem_algorithms.material.FourFiveDiethyloctane'>, 0.0011840664043430795, 'mol'], 'NaCl': [<class 'chemistrylab.chem_algorithms.material.NaCl'>, 0.01721152278447229, 'mol']}
    _____________
    {}
    _____________
    {}
    _____________



```python
analysis_action = np.array([3, 0, 0, 0])
lab.step(analysis_action)
```
    (0,
     array([4.62612025e-14, 4.08127308e-13, 3.21834031e-12, 2.26844550e-11,
            1.42916526e-10, 8.04815825e-10, 4.05105727e-09, 1.82263680e-08,
            7.32976062e-08, 2.63474618e-07, 8.46538342e-07, 2.43116074e-06,
            6.24078484e-06, 1.43193520e-05, 2.93674802e-05, 5.38355125e-05,
            8.82124514e-05, 1.29196211e-04, 1.69133069e-04, 1.97909278e-04,
            2.06996323e-04, 1.93516476e-04, 1.61708231e-04, 1.20782832e-04,
            8.06376411e-05, 4.81210409e-05, 2.56711428e-05, 1.22550327e-05,
            5.28547844e-06, 2.23699385e-06, 1.47232515e-06, 2.46629793e-06,
            5.79384141e-06, 1.30320877e-05, 2.64044793e-05, 4.78588809e-05,
            7.75433364e-05, 1.12302536e-04, 1.45376558e-04, 1.68212326e-04,
            1.73972352e-04, 1.60827956e-04, 1.32892819e-04, 9.81523699e-05,
            6.47977286e-05, 3.82375365e-05, 2.01737676e-05, 9.53585641e-06,
            4.11650444e-06, 1.89537570e-06, 1.72150385e-06, 3.56033024e-06,
            8.59383817e-06, 1.91984200e-05, 3.84822415e-05, 6.89751396e-05,
            1.10510337e-04, 1.58261144e-04, 2.02584110e-04, 2.31790022e-04,
            2.37051703e-04, 2.16695757e-04, 1.77058217e-04, 1.29312684e-04,
            8.44160750e-05, 4.92570289e-05, 2.56902622e-05, 1.19764827e-05,
            4.99055750e-06, 1.85877946e-06, 6.18817239e-07, 1.84144554e-07,
            4.89795475e-08, 1.16447136e-08, 2.47458742e-09, 4.70035344e-10,
            7.98033584e-11, 1.21107742e-11, 1.64278411e-12, 1.99179188e-13,
            2.15859011e-14, 2.09101349e-15, 1.81051016e-16, 1.40119185e-17,
            9.69306268e-19, 5.99354134e-20, 3.31254230e-21, 1.63641836e-22,
            7.22592803e-24, 2.85201641e-25, 1.00615580e-26, 3.17279192e-28,
            8.94268781e-30, 2.25298186e-31, 5.07353721e-33, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 6.45949145e-31, 2.83651654e-29, 1.11328977e-27,
            3.90578254e-26, 1.22480349e-24, 3.43297877e-23, 8.60095979e-22,
            1.92611074e-20, 3.85531203e-19, 6.89786864e-18, 1.10309213e-16,
            1.57682224e-15, 2.01470073e-14, 2.30084072e-13, 2.34872343e-12,
            2.14307322e-11, 1.74778969e-10, 1.27412414e-09, 8.30199642e-09,
            4.83528915e-08, 2.51721758e-07, 1.17130287e-06, 4.87174839e-06,
            1.81113955e-05, 6.01845386e-05, 1.78762348e-04, 4.74592700e-04,
            1.12623652e-03, 2.38889246e-03, 4.52917581e-03, 7.67544750e-03,
            1.16263861e-02, 1.57415215e-02, 1.90505292e-02, 2.06075162e-02,
            1.99252348e-02, 1.72202569e-02, 1.33025786e-02, 9.18521732e-03,
            5.66897122e-03, 3.12734093e-03, 1.54207193e-03, 6.79668214e-04,
            2.67758878e-04, 9.42876359e-05, 2.96769413e-05, 8.34914499e-06,
            2.09956806e-06, 4.71922476e-07, 9.48132239e-08, 1.70268919e-08,
            2.73307155e-09, 3.92135435e-10, 5.02885476e-11, 5.76448394e-12,
            5.90640438e-13, 5.40918594e-14, 4.42805786e-15, 3.23995820e-16,
            2.11896558e-17, 1.23874774e-18, 6.47271673e-20, 3.02304767e-21,
            1.26206474e-22, 4.70935366e-24, 1.57078452e-25, 4.68284204e-27,
            1.24785792e-28, 2.97233734e-30, 6.32807106e-32, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           dtype=float32),
     False)



Lastly, we use the characterization bench. In this case, we are going to perform an absorption spectra analysis on our vessel that we get back. This is designed so that the agent can identify what is in the vessel without explicitly telling the agent. That's all for this part of the tutorial on the Lab environment, next we will cover the Lab Manager wrapper for the lab environment. 

## Lab Manager

The Lab Manager at the moment doesn't support training agents, however, it does support pre-trained or heuristic agents, or even a human agent.


```python
from chemistrylab.lab.manager import Manager
```


```python
manager = Manager()
manager.agents
```




    {'random': chemistrylab.lab.agent.RandomAgent}



The output above shows us what agents are available to run the Manager environment. You can also make your own custom agents using our agent api.


```python
from chemistrylab.lab.agent import Agent
```


```python
class CustomAgent(Agent):
    def __init__(self):
        self.name = 'custom_agent'
        self.prev_actions = []

    def run_step(self, env, spectra):
        """
        this is the function where the operation of your model or heuristic agent is defined
        """
        action = np.array([])
        if len(self.prev_actions) == 0:
            action = np.array([0, 0, 0, 0])
        elif self.prev_actions[-1][0] == 0:
            action = np.array([1, 0, 0, 0])
        elif self.prev_actions[-1][0] == 1:
            action = np.array([2, 0, 2, 0])
        elif self.prev_actions[-1][0] == 2:
            action = np.array([3, 0, 2, 0])
        elif self.prev_actions[-1][0] == 3:
            action = np.array([4, 0, 0, 0])
        else:
            assert False
        self.prev_actions.append(action)
        return action
```


```python
custom_agent = CustomAgent
```


```python
# Now that we have created a custom agent to run the whole lab process for us we need to register it
# with our environment
manager.register_agent('custom_agent', custom_agent)
```


```python
# Now that the agent has been registered we change the mode to the name of the new agent so we will run the manager
# with the new agent
manager.mode = 'custom_agent'
```


```python
manager.run()
```

    WurtzReact-v1
    WurtzExtract-v1
    Distillation-v0


## Closing Remarks
And the api is just as simple as that! We hope this has been informative and you should now be able to run the Lab and Lab Manager smoothly.
