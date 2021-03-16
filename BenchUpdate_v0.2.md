# Reaction Bench Changes - Version 0.1

## Markdown File Detailing Changes to the Reaction Bench

### Associated Issues: 98, 100, 101, 102, 103, 104, 105

### Changes:

- Vessel:
    - Fixed non-existant pressure issue.
- Reaction Bench v1:
    - Added the reaction parameter, time step, number of steps, and reaction file identifier parameters.
- Reaction Bench v1 Engine:
    - Implemented handling of input materials and solutes into the initialization vessel.
    - Minimized the interaction between the bench engine and key reactino base parameters. Instead, these parameters should be found in the vessel itself.
    - Added the reaction file identifier to the validate parameters method.
    - Fixed error in reset function where the vessel's Tmax property was not called.
- Reaction Base:
    - Added Pylinting for errors and warnings.
    - Changed the _find_reaction_file method such that the available reactions directory is findable from any directory.
    - Clipped the reaction filename such that it can be found without the need to specify the file extension.
    - Fixed issue where the reaction file parameters were incorrectly acquired.
    - Fixed unit specification for setting the volume parameters when reseting the input vessel in reset.
    - Fixed error in get_conc_change where the wrong array was being iterated through.
    - Added preliminary update_vessel documentation.
    - Fixed iteration variable warning and incorrect volume method in perform_action.
    - Added concentration array to plot_graph to eliminate error message.

Changes Still to Be Made:
- Implement plotting.
- Add documentation throughout.
- Add additional means by which the reaction file can be properly specified.