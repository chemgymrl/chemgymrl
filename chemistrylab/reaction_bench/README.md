# ODEworld
OpenAI Gym framework environment for controlling reaction parameters for reactions modelled by rate law ODEs.

## To install:
```bash
pip install -e ./
```

## Environments

Version | Description
--- | ---
v0 | State is absorption spectra, current time, temperature, volume, pressure, and amount of reactants left in hand. Change in temperature and volume, and amount of reactant to add are actions. Change in amount of desired product is proportional to reward. Negative reward is applied for trying to add more reactant than you have.
ODEWorld\_0-v0 | Chemical reaction 0 modelled by ODEs.
ODEWorld\_0\_overlap-v0 | Chemical reaction 0 modelled by ODEs. Spectra peaks for each species have more overlap in this environment.
ODEWorld\_1-v0 | Chemical reaction 1 modelled by ODEs.
ODEWorld\_1\_overlap-v0 | Chemical reaction 1 modelled by ODEs. Spectra peaks for each species have more overlap in this environment.
ODEWorld\_2-v0 | Chemical reaction 2 modelled by ODEs.
ODEWorld\_2\_overlap-v0 | Chemical reaction 2 modelled by ODEs. Spectra peaks for each species have more overlap in this environment.
ODEWorld\_3-v0 | Chemical reaction 3 modelled by ODEs.
ODEWorld\_3\_overlap-v0 | Chemical reaction 3 modelled by ODEs. Spectra peaks for each species have more overlap in this environment.
ODEWorld\_4-v0 | Chemical reaction 4 modelled by ODEs.
ODEWorld\_4\_overlap-v0 | Chemical reaction 4 modelled by ODEs. Spectra peaks for each species have more overlap in this environment.
ODEWorld\_5-v0 | Chemical reaction 5 modelled by ODEs.
ODEWorld\_5\_overlap-v0 | Chemical reaction 5 modelled by ODEs. Spectra peaks for each species have more overlap in this environment.
ODEWorld\_6-v0 | Chemical reaction 6 modelled by ODEs.
ODEWorld\_6\_overlap-v0 | Chemical reaction 6 modelled by ODEs. Spectra peaks for each species have more overlap in this environment.

## Reactions
\|X\| means that X is the desired product.

ID in v0 | Description
--- | ---
Reaction 0 | A + B -> \|C\|: Designed to introduce reactions.
Reaction 1 | A <-> B, B -> \|C\|: Designed to introduce reversible reactions.
Reaction 2 | A + B -> D, A + C -> E, B + C -> F, D + E -> \|G\|: Designed to introduce useless reactions.
Reaction 3 | A + B + C -> E, A + D -> F, B + D -> G, C + D -> H, F + G + H -> \|I\|: Designed to introduce reagent management reactions.
Reaction 4 | A + B -> C, A + B -> D, D -> \|E\|: Designed to introduce kinetic vs thermodynamical favoured products.
Reaction 5 | A + B -> D, C + D -> E, E -> \|F\| + B: Designed to introduce catalyst reactions.
Reaction 6 | A + B -> 2C, C -> \|D\|, \|D\| -> C, A + \|D\| -> E, B + \|D\| -> F: Designed to introduce loseable product reactions.

## To run

See demo.py for an example.
