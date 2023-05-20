# [Materials](#Material)

# [Reactions](#Reaction)

# [Characterization Bench](#Characterization-Bench-Info)

# [Vessel](#Vessel-Info)

# [Shelf](#Shelf-Info)

# [General Benches](#General-Bench)

# Material


- Material objects now have a mol property (amount of material)
    - This allows for extensive properties to be included as properties
    - Material Dict no only stores references to material objects
- Added dictionary called `REGIRSTY` to store materials according to their name

### Example:
    
```python
from chemistrylab.chem_algorithms import material
H2O = material.REGISTRY["H2O"]()
H2O.mol = 1
print(H2O.litres)
print(H2O.heat_capacity)
```
```
> 0.018069207622868604
> 75.3261195
```


# Reaction

- Reaction info is now stored in a json file which is loaded into a ReactInfo named tuple
    ```python
    class ReactInfo(NamedTuple):
        name: str
        REACTANTS:        Tuple[str]
        PRODUCTS:         Tuple[str]
        SOLVENTS:         Tuple[str]
        MATERIALS:        Tuple[str]
        pre_exp_arr:      np.array
        activ_energy_arr: np.array
        stoich_coeff_arr: np.array
        conc_coeff_arr:   np.array
    ```
- All reactions are handled using `reactions.Reaction` which takes a ReactInfo tuple during `__init__`.
    - Reactions still use solve_ivp from scipy
    - A solver using newton's method is implemented as an alternative to solve_ivp (it uses variable step size for more accurate results)
    - Reactions interface with vessels through the method `update_concentrations(self, vessel)` which will update a vessel's material dictionary with new molar amounts for materials involved in the reaction

    
### Example Reaction json
```json
{
  "name": "equilibrium reactions",
  "REACTANTS": ["CuS04*5H2O", "CuS04", "H2O"],
  "PRODUCTS":  ["CuS04*5H2O", "CuS04", "H2O"],
  "SOLVENTS":  ["H2O"],
  "MATERIALS": ["CuS04*5H2O", "CuS04", "H2O"],
  "pre_exp_arr": [ 1.0, 1.0  ],
  "activ_energy_arr": [
    [ 8.0  ],
    [ 10.0 ]
  ],
  "stoich_coeff_arr": [
    [ 1.0, 1.0, 0.0 ],
    [ 0.0, 0.0, 1.0 ]
  ],
  "conc_coeff_arr": [
    [-1.0, 1.0 ],
    [ 1.0, 1.0 ],
    [ 5.0,-1.0 ]
  ]
}

```

 # Characterization Bench Info
 
- Currently handles all observations
- Has a dictionary of (name,function) pairs so characterization methods can be passed in as a string
    - A list of of strings specifying characterization methods are passed along in `__init__`
    - `get_observation` or `__call__` is used to get a concatenation of all characterization methods.
    
### Example:

```python
from chemistrylab.benches.characterization_bench import CharacterizationBench
from chemistrylab.chem_algorithms import vessel,material

v = vessel.Vessel("A")
H2O = material.REGISTRY["H2O"]()
H2O.mol = 1
v.material_dict = {H2O._name:H2O}
char1 = CharacterizationBench(["PVT","targets" ], targets = ["H2O","NaCl"], n_vessels = 1)
char2 = CharacterizationBench(["PVT"], targets = ["H2O","NaCl"], n_vessels = 1)

print(char1([v],"NaCl"))
print(char2([v],"NaCl"))

```

```
> [[0.29699999 0.01806921 0.         0.         1.        ]]
> [[0.29699999 0.01806921 0.        ]]
```

### Future Ideas

- Make a characterization interface which multiple different benches inherit from

# Vessel Info
 
- Similar functionality to previous versions. Vessels are used to store materials via material_dict and solute_dict.
- Also has some general properties like volume, heat capacity, temperature, etc.
- The state of a vessel is changed using an event:

    ```python
    class Event(NamedTuple):
        name: str
        parameter: tuple
        other_vessel: Optional[object]
    ```
- Events are passed into `the push_event_to_queue()` function and executed in the order they are listed.
    - List of currently supported events:
        1. pour by volume
        2. pour by percent
        3. drain by pixel
        4. mix
        5. update layer
        6. heat contact
 
 ### Example:
 
```python

from chemistrylab.chem_algorithms import vessel,material
v=vessel.Vessel("A")
H2O = material.H2O(mol=1)
Na = material.Na(mol=1)
Na.phase="l"
Na.polarity = 2.0
Cl = material.Cl(mol=1)
C6H14 = material.C6H14(mol=1)
ether=material.DiEthylEther(mol=0.2)
v.material_dict={str(Na):Na,str(Cl):Cl,str(C6H14):C6H14,str(ether):ether}
v.validate_solvents()
v.validate_solutes()

print(v.get_material_dataframe(),"\n")
print(v.get_solute_dataframe(),"\n")

event = vessel.Event("mix",(0.5,),None)
v.push_event_to_queue([event],0)
print(v.get_solute_dataframe())

```

```
>                Amount Phase  Solute  Solvent
> Na                1.0     l    True    False
> Cl                1.0     l    True    False
> C6H14             1.0     l   False     True
> diethyl ether     0.2     l   False     True 
> 
>        C6H14  diethyl ether
> Na  0.833333       0.166667
> Cl  0.833333       0.166667 
> 
>        C6H14  diethyl ether
> Na  0.689072       0.310928
> Cl  0.689072       0.310928
```

# Shelf Info

Currently, the shelf is an array-like structure to store vessels.

One important difference is that shelves have a reset function to return the set of vessels to it's original state.

```python

from chemistrylab.lab.shelf import Shelf
from chemistrylab.chem_algorithms.vessel import Vessel

shelf = Shelf([  
    Vessel("Beaker"),
    Vessel("Test Tube"),
    Vessel("Waste Bin"),
], n_working = 2)

print(shelf)
shelf.append(Vessel("New Vessel"))
print(shelf)
shelf.reset()
print(shelf)

```


```
> Shelf: (Beaker, Test Tube, Waste Bin)
> Shelf: (Beaker, Test Tube, Waste Bin, New Vessel)
> Shelf: (Beaker, Test Tube, Waste Bin)
```

# General Bench

- Uses the Characterization Bench for observations and actions correspond to Events for vessels.
- Actions are parameterized with an Action Named tuple
    - One Action object can correspond to multiple discrete actions  (all with the same event)
    - It can also correspond to one dimension of a continuous action (an event with a continuous parameter)
    
```python
class Action(NamedTuple):
    vessels: Tuple[int]
    parameters: Tuple[tuple]
    event_name: str
    affected_vessels: Optional[Tuple[int]]
    dt: float
    terminal: bool
         
class ContinuousParam(NamedTuple):
    min_val: float
    max_val: float
    thresh: float
    other: object
```

- Uses a Reaction to update the contents of each vessel when specified.



### Example:

```python
class GeneralWurtzExtract_v2(GenBench):
    """
    Class to define an environment which performs a Wurtz extraction on materials in a vessel.
    """

    def __init__(self):
        e_rew= RewardGenerator(use_purity=True,exclude_solvents=True,include_dissolved=True)
        shelf = VariableShelf( [
            lambda x:wurtz_vessel(x)[0],
            lambda x:vessel.Vessel("Beaker 1"),
            lambda x:vessel.Vessel("Beaker 2"),
            lambda x:make_solvent("C6H14"),
            lambda x:make_solvent("diethyl ether")
        ],[], n_working = 3)
        amounts=np.linspace(0.2,1,5).reshape([5,1])
        pixels = (amounts*10).astype(np.int32)
        actions = [
            Action([0], pixels,              'drain by pixel',[1],  0.01, False),
            Action([0],-amounts,             'mix',           None, 0.01, False),
            Action([1], amounts,             'pour by volume',[0],  0.01, False),
            Action([2], amounts,             'pour by volume',[0],  0.01, False),
            Action([0], amounts,             'pour by volume',[2],  0.01, False),
            #If pouring by volume takes time, then there is no change in observation when waiting after pouring in some cases
            Action([3], amounts/2,           'pour by volume',[0],  0,    False),
            Action([4], amounts/2,           'pour by volume',[0],  0,    False),
            Action([0,1,2], 32**amounts/200, 'mix',           None, 0,    False),
            Action([0], [[0]],               'mix',           None, 0,    True)
        ]
        
        react_info = ReactInfo.from_json(REACTION_PATH+"/chloro_wurtz.json")

        super(GeneralWurtzExtract_v2, self).__init__(
            shelf,
            actions,
            react_info,
            ["layers","targets"],
            reward_function=e_rew
        )

bench = GeneralWurtzExtract_v2()
print(bench.observation_space)
print(bench.action_space)
```

```
> Box(0.0, 1.0, (3, 107), float32)
> Discrete(41)
```


```python

```
