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
- All reactions are handled using `reactions.Reaction` which takes a ReactInfo tuple during __init__.
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
