# ExtractWorld
OpenAI Gym framework environment for performing extractions

## Engines:
Version|Description
---|---
extractworld_v1_engine|An generalized engine to handle extractions using chemistrylab API, and the render function returns the gaussian distribution for materials in vessels, and the pixel representations for vessels.

## Extractions:
Version|Description
---|---
water_oil_v1|Use C6H14 to extract Na and Cl dissolved in H2O in ExV 
wurtz_v0|Use ethoxyethane to extract dodecane into the ExV

## Environments:
Version|Extraction|Engine|Description
---|---|---|---
ExtractWorld_0-v1|water_oil_v1|extractworld_v1_engine|An environment that handles extraction of Na- or Cl+ from H2O using C6H14. 

## ExtractWorld_0-v1

### Action space:
action: [action index, action parameter]

Action Index|Description|Parameters
---|---|---
0|Valve, draining from ExV to Beaker1|speed multiplier, relative to max_valve_speed
1|Mix ExV|mixing coefficient, *-1 when passed into mix function
2|Mix B1|mixing coefficient, *-1 when passed into mix function
3|Mix B2|mixing coefficient, *-1 when passed into mix function
4|Pour from B1 to ExV|volume multiplier, relative to max_vessel_volume
5|Pour from B1 to B2|volume multiplier, relative to max_vessel_volume
6|Pour from ExV to B2|volume multiplier, relative to max_vessel_volume
7|Add oil, pour from Oil Vessel to ExV|volume multiplier, relative to max_vessel_volume
8|Done|Parameter doesn't matter


### State:
*The state for the environment is a list containing states of each vessel*

Each vessel has three matrices to represent its state:

[material_dict_matrix, solute_dict_matrix, layer_vector]

 
**material_dict_matrix (number of available material * 10):**

-|Density|Polarity|Temperature|Pressure|Solid Flag|Liquid Flag|Gas Flag|Charge|Molar Mass|Amount
---|---|---|---|---|---|---|---|---|---|---
Air|
H2O|
H|
H2|
.|
.|
.|

**solute_dict_matrix (number of available material * number of available material):**

Each cell means how much of that solute (row) dissolved in that solvent (column)

-|Air|H2O|H|H2|.|.|.
---|---|---|---|---|---|---|----
Air|
H2O|
H|
H2|
.|
.|
.|

**layer_vector (1 * number of pixel):**

acquired form vessel.get_layers()

### Explain the graph:

Each row has two figure to show information of one vessel.

The left one shows the gaussian distribution of layers of the vessel.

The right one is the pixel (layers) representation of the vessel.

## To install:
```bash
pip install ./
```

