# Reaction Bench Changes - Version 0.3

## Markdown File Detailing Changes to the Reaction Bench

### Associated Issues: 98, 100, 101, 102, 103, 104, 105

### Changes:

- Chloro Wurtz:
    - Fixed all thermodynamic variables to floating point.
- Reaction Bench v1:
    - Changed the inputted list of solutes, such that the H2O provided is listed as a material rather than a solute because it is a material with the unique property that it can be a solute.
- Reaction Bench v1 Engine:
    - Added `_prepare_materials` and `_prepare_vessel` methods to prepare the vessel upon initialization. Both methods of providing a list of materials and solutes or providing the path to an existing reaction vessel continue to be supported.
- Reaction Base:
    - Added documentation for all methods (except plotting).
    - Moved initialization of the (empty) `n` array to the constructor class method.
    - Moved initialization of the `initial_in_hand` and `initial_solutes` parameters to the constructor class method.
    - Implemented flexible acquisition of parameters in `_get_reaction_params`.
    - Added `vessel_deconstruct` method to create the `n` array uing the inputted vessel to `perform_action`. This contrasts the persistent `n` array that is maintained in `reaction_base` between steps and only takes input from the reaction vessel upon a reset of the reactino environment.
    - Added initialization of the `n` array in the `reset` function. Upon `reaction_base` initialization, the `n` array is populated by the initial materials acquired from the initialized reaction vessel.
    - Added `vessel_deconstruct` to `perform_action` to acquire the vessel temperature and volume immediately upon recieving the reaction vessel and modify the `n` array to contain the proper material amounts as given by the reaction vessel.

Changes Still to Be Made:
- Implement flexible material and solute preparation in `reaction_bench_v1_engine`. Currently, it is fixed on the list having two parameters labelled: "Material" and "Initial".
