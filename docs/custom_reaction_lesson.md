# Creating a Custom Reaction

In this tutorial, I am going to walk you through how reactions work and how to make your own custom reaction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chemgymrl/chemgymrl/blob/main/lessons/notebooks/custom_reaction.ipynb)

```python
from chemistrylab.reactions.reaction_info import ReactInfo
from chemistrylab.reactions.reaction import Reaction
from chemistrylab import material,vessel

import numpy as np
from IPython.display import display,clear_output,JSON

```

## Adding hydronium and hydroxide materials


```python
class H3O(material.Material):
    def __init__(self, mol=0):
        super().__init__(
            mol=mol,
            name='H3O',
            density={'s': None, 'l': 0.997, 'g': None},
            polarity=abs(2 * 1.24 * np.cos((109.5 / 2) * (np.pi / 180.0))),
            temperature=298,
            pressure=1,
            phase='l',
            molar_mass=19.0,
            color=0.2,
            charge=0.0,
            boiling_point=373.15,
            solute=True,
            specific_heat=4.1813,
            enthalpy_vapor=40650.0,
            index=1
        )


class OH(material.Material):
    def __init__(self, mol=0):
        super().__init__(
            mol=mol,
            name='OH',
            density={'s': None, 'l': 0.997, 'g': None},
            polarity=abs(2 * 1.24 * np.cos((109.5 / 2) * (np.pi / 180.0))),
            temperature=298,
            pressure=1,
            phase='l',
            molar_mass=19.0,
            color=0.2,
            charge=0.0,
            boiling_point=373.15,
            solute=True,
            specific_heat=4.1813,
            enthalpy_vapor=40650.0,
            index=3
        )
        
material.register(H3O,OH)
```

## Setting the reactants and products

$H_2O$, $H_3O^+$, and  $OH^-$ are all both reactants and products (since we are including both the forward and reverse reaction)


```python
name="Autoionization"
REACTANTS = ["H2O","H3O","OH"]
PRODUCTS = ["H2O","H3O","OH"]
SOLVENTS = ["H2O"]
MATERIALS=["H2O","H3O","OH"]
```

## Setting rates

For each reaction we have: $k = Ae^{\frac{Ea}{RT}}$

To set this we know [$OH^-$][$H_3O^+$] = $1\cdot 10^{-14}$ at equilibrium.

Additionally we know [H_2O] is always 55.34



```python
# 55.34 is the concentration of water in water and 1e-14 is Keq in the autoionization reaction
pre_exp_arr = np.array([55.34,1e-14])*1e7 
# No idea what the activation energies are
activ_energy_arr = np.array([1.0,1.0])
```

## Setting Stoicheometry coefficients

This will be a [reactions, reactants] shape array


```python
stoich_coeff_arr = np.array([
    [0, 1, 1], # H3O + OH -> H2O+H2O
    [1, 0, 0] # H2O + H2O -> H3O + OH
]).astype(np.float32)
```

## Setting concentration coefficients

This will be a [materials, reactions] shape array. It represents the change in concentrations given by each reaction. (Changes in concentration will always be within the column space of this matrix)


```python

conc_coeff_arr = np.array([
    [2, -2],
    [-1, 1],
    [-1, 1]
]).astype(np.float32)

info = ReactInfo(name,REACTANTS,PRODUCTS,SOLVENTS,MATERIALS,pre_exp_arr,activ_energy_arr,stoich_coeff_arr,conc_coeff_arr)
```

## Setting up the reaction


```python
reaction = Reaction(info)
v = vessel.Vessel("Water Vessel")
H2O = material.H2O(mol=1)
v.material_dict = {H2O._name:H2O}
v.default_dt=0.1
```


```python
v.get_material_dataframe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amount</th>
      <th>Phase</th>
      <th>Solute</th>
      <th>Solvent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>H2O</th>
      <td>1</td>
      <td>l</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
reaction.update_concentrations(v)
v.get_material_dataframe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Amount</th>
      <th>Phase</th>
      <th>Solute</th>
      <th>Solvent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>H2O</th>
      <td>1.000000e+00</td>
      <td>l</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>H3O</th>
      <td>1.804012e-09</td>
      <td>l</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>OH</th>
      <td>1.804012e-09</td>
      <td>l</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



## Looking at the pH


```python
pH = -np.log(v.material_dict["OH"].mol/v.material_dict["H2O"].litres)/np.log(10)

print(f"pH: {pH}")
```

    pH: 7.000699771831425
    

## Saving as a json file


```python
info.dump_to_json("autoionization.json")
json_text = "".join(line for line in open("autoionization.json","r"))
print(json_text)
```

    {
      "name": "Autoionization",
      "REACTANTS": [
        "H2O",
        "H3O",
        "OH"
      ],
      "PRODUCTS": [
        "H2O",
        "H3O",
        "OH"
      ],
      "SOLVENTS": [
        "H2O"
      ],
      "MATERIALS": [
        "H2O",
        "H3O",
        "OH"
      ],
      "pre_exp_arr": [ 553400000.0, 1e-07  ],
      "activ_energy_arr": [ 1.0, 1.0  ],
      "stoich_coeff_arr": [
        [ 0.0, 1.0, 1.0 ],
        [ 1.0, 0.0, 0.0 ]
      ],
      "conc_coeff_arr": [
        [ 2.0,-2.0 ],
        [-1.0, 1.0 ],
        [-1.0, 1.0 ]
      ]
    }
    

## Loading from your json file


```python
info_copy = ReactInfo.from_json("autoionization.json")

print(info_copy.name)
```

    Autoionization
    
