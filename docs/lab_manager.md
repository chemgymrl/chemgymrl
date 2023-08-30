[chemgymrl.com](https://chemgymrl.com/)

# Lab Manager

<span style="display:block;text-align:center">![Lab Manager](tutorial_figures/lab_manager.png)

The lab manager acts as another agent in the ChemGymRL environment, yet the lab manager’s task is to organize and dictate the activity of bench agents, input and output vessels, benches, and bench environments.
 

A useful analogy to describe the role of the lab manager in the ChemGymRL environment is as a lab instructor or lab director in a professional lab environment. The lab instructor oversees operation over the entire laboratory, yet does not perform bench experiments themselves, leaving such experimentation to their subordinates, the bench agents. The lab instructor also interacts with exterior clientele taking into account the task to be completed and allocating resources and personnel accordingly.
 

Much like bench agents, the lab manager is also learning how to best perform tasks and achieve desired outcomes by means of positive and negative reinforcement in the form of rewards. Bench agents interact with the lab manager by receiving instruction and returning the results of experiments. The lab manager interacts with the shelf, the container for all vessels not currently in use, as well as the results of the characterization bench to allocate vessels to benches and bench agents.

## Input

A lab manager's configuration is handled by a json file. Here, the benches, policies, and actions available to the manager are specified. The json below configures the wurtz lab manager.

```json
{
  "benches": [
    ["Reaction",     "GenWurtzReact-v2"],
    ["Distillation", "GenWurtzDistill-v2"],
    ["Extraction",   "GenWurtzExtract-v2"]
  ],
  "policies": [
    ["Heuristic", "chemistrylab.lab.heuristics:WurtzReactHeuristic",      {},                   0],
    ["Heuristic", "chemistrylab.lab.heuristics:GenWurtzDistillHeuristic", {"heat_act": 9 ...},  1],
    ["Heuristic", "chemistrylab.lab.heuristics:GenWurtzExtractHeuristic", {"drain_act": 1 ...}, 2]
  ],
  "targets": [
    "CCCCCCCCCCCC",
    "CCCCCCC(C)CCCC",
    "CCCCCCC(CC)CCC",
    "CCCCC(C)C(C)CCCC",
    "CCCCC(C)C(CC)CCC",
    "CCCC(CC)C(CC)CCC",
    "[Na+].[Cl-]"
  ],
  "actions": [
    ["end experiment",{},                                0.0, null],
    ["swap vessels",  {"bench_idx": -1,"vessel_idx": 0}, 0.0, null],
    ["swap vessels",  {"bench_idx": -1,"vessel_idx": 1}, 0.0, null],
    ["swap vessels",  {"bench_idx": -1,"vessel_idx": 2}, 0.0, null],
    ["characterize",  {"observation_list": ["PVT"]},     0.0, null],
    ["run bench",     {"bench_idx": 0,"policy_idx": 0},  0.0, 0   ],
    ["swap vessels",  {"bench_idx": 0,"vessel_idx": 0},  0.0, 0   ],
    ["run bench",     {"bench_idx": 1,"policy_idx": 0},  0.0, 1   ],
    ["swap vessels",  {"bench_idx": 1,"vessel_idx": 0},  0.0, 1   ],
    ["run bench",     {"bench_idx": 2,"policy_idx": 0},  0.0, 2   ],
    ["swap vessels",  {"bench_idx": 2,"vessel_idx": 0},  0.0, 2   ]
  ]
}
```

The above json file gives a list of ordered parameters for the construction of Benches, Policies, and Actions. The corresponding named tuples to these parameters lists are:

<table>
<tr>
<td>
  
```python
class BenchParam(NamedTuple):
    name: str
    gym_id: str
```  
</td>
<td>

```python
class PolicyParam(NamedTuple):
    name: str
    entry_point: str
    args: dict
    bench_idx: int
```

</td>
<td>

```python
class ManagerAction(NamedTuple):
    name: str
    args: dict
    cost: float
    bench: Optional[int]
```
</td>

</tr>
</table>

## Output

The lab manager itself does not have any specific outputs. We can, however, use a gui to see what the lab manager does by running the command:

```
python -m chemistrylab.lab.manager
```


