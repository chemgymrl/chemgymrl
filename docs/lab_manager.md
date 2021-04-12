[chemgymrl.com](https://chemgymrl.com/)

## Lab Manager

<span style="display:block;text-align:center">![Lab Manager](tutorial_figures/lab_manager.png)

The lab manager acts as another agent in the ChemGymRL environment, yet the lab manager’s task is to organize and dictate the activity of bench agents, input and output vessels, benches, and bench environments.
 

A useful analogy to describe the role of the lab manager in the ChemGymRL environment is as a lab instructor or lab director in a professional lab environment. The lab instructor oversees operation over the entire laboratory, yet does not perform bench experiments themselves, leaving such experimentation to their subordinates, the bench agents. The lab instructor also interacts with exterior clientele taking into account the task to be completed and allocating resources and personnel accordingly.
 

Much like bench agents, the lab manager is also learning how to best perform tasks and achieve desired outcomes by means of positive and negative reinforcement in the form of rewards. Bench agents interact with the lab manager by receiving instruction and returning the results of experiments. The lab manager interacts with the shelf, the container for all vessels not currently in use, as well as the results of the characterization bench to allocate vessels to benches and bench agents.

## Input

The lab manager's input are the three main environments that it will be working with, mainly the reaction, extraction,
and distillation environments. This is made available and specified under `manager_v1.py`.

![lab manager input](../tutorial_figures/labmanager/lab_input.png)

## Output

The lab manager's output will be messages from the environment that it is currently performing actions on. It will also 
depend on the action that is being performed. The lab manager itself does not have any specific outputs. We can, 
however, use lab manager to access the extraction environment and get outputs from that.

![extract action](../tutorial_figures/labmanager/extract_output.png)
