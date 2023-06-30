[chemgymrl.com](https://chemgymrl.com/)

# Lab Manager

<span style="display:block;text-align:center">![Lab Manager](tutorial_figures/lab_manager.png)

The lab manager acts as another agent in the ChemGymRL environment, yet the lab manager’s task is to organize and dictate the activity of bench agents, input and output vessels, benches, and bench environments.
 

A useful analogy to describe the role of the lab manager in the ChemGymRL environment is as a lab instructor or lab director in a professional lab environment. The lab instructor oversees operation over the entire laboratory, yet does not perform bench experiments themselves, leaving such experimentation to their subordinates, the bench agents. The lab instructor also interacts with exterior clientele taking into account the task to be completed and allocating resources and personnel accordingly.
 

Much like bench agents, the lab manager is also learning how to best perform tasks and achieve desired outcomes by means of positive and negative reinforcement in the form of rewards. Bench agents interact with the lab manager by receiving instruction and returning the results of experiments. The lab manager interacts with the shelf, the container for all vessels not currently in use, as well as the results of the characterization bench to allocate vessels to benches and bench agents.

## Input

The lab manager's input are the three main environments that it will be working with, mainly the reaction, extraction,
and distillation environments. This is made available and specified under `lab.py`.

```python
class Lab(gym.Env, ABC):
    """
    The lab class is meant to be a gym environment so that an agent can figure out how to synthesize different chemicals
    """
    def __init__(self, render_mode: str = None, max_num_vessels: int = 100):
        """

        Parameters
        ----------
        render_mode: a string for the render mode of the environment if the user wishes to see outputs from the benches
        max_num_vessels: the maximum number of vessels that the shelf can store
        """
        all_envs = envs.registry.all()
        # the following parameters list out all available reactions, extractions and distillations that the agent can use
        self.reactions = [env_spec.id for env_spec in all_envs if 'React' in env_spec.id]
        self.extractions = [env_spec.id for env_spec in all_envs if 'Extract' in env_spec.id]
        self.distillations = [env_spec.id for env_spec in all_envs if 'Distill' in env_spec.id]
        self.characterization = list(CharacterizationBench().techniques.keys())
        self.characterization_bench = CharacterizationBench()
```

## Output

The lab manager's output will be messages from the environment that it is currently performing actions on. It will also 
depend on the action that is being performed. The lab manager itself does not have any specific outputs. We can, 
however, use lab manager to access the extraction environment and get outputs from that.

```
>>> python manager.py
Index: Action
0: load vessel from pickle
1: load distillation bench
2: load reaction bench
3: load extraction bench
4: load characterization bench
5: list vessels
6: create new vessel
7: save vessel
8: quit
```
