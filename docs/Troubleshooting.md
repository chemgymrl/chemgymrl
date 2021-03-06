[chemgymrl.com](https://chemgymrl.com/)

# Troubleshooting

You may run into some unexpected errors when running chemgymrl environments from time to time. 

Here we will discuss some possible error messages here and try to find a solution to them.

## Errors

**Not adding an entrypoint when registering an environment**

![registration error entrypoint](../tutorial_figures/troubleshooting/registration_error_entrypoint.png)

![registration error](../tutorial_figures/troubleshooting/registration_error.png)

When registering any environment it's important to note that the corresponding entry point must also be added. In the 
picture above the entry point to `WurtzReact-v1` was expected to be the class `ReactionBenchEnv_0` in 
`reaction_bench_v1.py`. Failure to add the corresponding class will result in this error message.

**Invalid vessel acquisition**

![invalid vessel input](../tutorial_figures/troubleshooting/invalid_vessel_input.png)

When initializing environments that need an input vessel such as extraction or distillation, the path to the vessel must 
be specified or generated (as is the case of oil_vessel in `extract_bench_v1.py`). Failure to ensure at least one of the
methods specified will result in the following error message.

**Drain function error - Extraction**

When pouring by volume in extraction from any vessel to the extraction vessel, a second instance of running the 
fully_mix function will be needed for the vessel to properly update.

This leads to an error with seperate.map_to_state when running fully_mix for pouring by volume between ancillary vessels or pouring the extraction 
vessel into an ancillary vessel.

The current state of the code has fully_mix running twice in all extractions, but only for the pouring of an ancillary 
vessel into the extraction vessel.