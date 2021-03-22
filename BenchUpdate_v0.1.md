# Reaction Bench Changes - Version 0.1

## Markdown File Detailing Changes to the Reaction Bench

### Associated Issues: 98, 100, 101, 102, 103, 104, 105

### Changes:

- Available Reactions Directory Created
- Chloro-Wurtz Reaction File Created
- Reaction Base:
    - Eliminated Reaction class initialization parameters as these were to be specified by the reaction file or implemented as part of a Reaction class method.
    - Created methods for obtaining and retrieving information from the specified reaction file. These are `_find_reaction_file` and `_get_reaction_params`, respectively.
    - Unpacked the reaction parameters and made them available to all methods
    - Changed the `initial_in_hand` and `cur_in_hand` variables such that they are given the proper values in `reset` rather than `__init__`.
    - Moved the `get_pressure` and `get_part_pressure` methods from the reaction class to the vessel as they are supposed to be vessel functions.
    - Used the reset function to reset the vessel parameters to the initial values specified in the reaction file.
    - Created the `get_reaction_constants` method which takes the vessel temperature and the array of concentrations to calculate an array containing the rate constants for each reaction.
    - Created the `get_rates` method which takes the rate constant array and the concentrations array to calculate the rates of each reaction set to occur.
    - Created the `get_conc_change` method which takes the rates array, the concentration array, and the time-step to calculate the changes in concentration of each material involved in any reaction.
    - Implemented an `action_deconstruct` method to deconstruct the action immediately after it is recieved in the `perform_action` method.
    - Ensured the `update` function was using the `get_reaction_constants`, `get_rates`, and `get_conc_change` methods properly.
    - Renamed the `step` method to the `perform_action` method for consistency with other benches.
    - Included `action_deconstruct` in the `perform_action` method and ensured the proper parameters were being passed between `perform_action` and `update`.
    - Moved the plotting operations out of `perform_action` and into `plotting_step` which is to be called in one of the reaction bench's render functions.

- Reaction Bench v1 Engine:
    - Limited the initialization parameters to just the materials, solutes, vessel paths, number of steps, time-step, and overlap. The other thermodynamic variables, which used to be specified upon initialization, are better specified by the reaction file (because they can be reaction specific).
    - Shortened the `_validate_parameters` method to account for the decrease in initialization parameters.
    - Ensured the vessel is provided to the reaction base's reset function to reset the vessel's thermodynamic parameters as well as the reaction base environment.
    - Removed the plotting parameters from the `step` function as these are better suited in a `render` function.


### Changes Still To Be Made:
- In `_find_reaction_file`, ensure the available reactions directory can be obtained from any folder (use the `reaction_base.py`'s local path)
- Add documentation to the `update_vessel` method in `reaction_base.py`.
- Fix the plotting operations in `reaction_base.py`, so that they can be properly called and operate in conjunction with the render functions in the reaction bench engine.
- Consider moving the reaction constants, rates, and concentration change functions to a separate file.
- Fix the plotting movements in reaction_base and reaction bench engine.
- Consider changing the behaviour of the n array in `reaction_base.py`, such that it is specified in the `perform_action` function by deconstructing the vessel and thus does not persist between steps. If this were implemented, the vessel deconstruction function would now become useful, otherwise, such a function is overkill for how much it would actually do.
- Implement a compatibility function in `reaction_base.py` to be called once a vessel has been passed to the reaction base in reset. This would test the compatibility with the incoming vessel from the reaction bench engine and the reactions that are requested to occur.
- General Testing of the Reaction Base Methods.
