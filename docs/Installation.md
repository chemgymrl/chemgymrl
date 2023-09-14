[chemgymrl.com](https://chemgymrl.com/)

# Installation

Installing the chemgymrl library can be done by installing straight from the [ChemGymRL github repository](https://github.com/chemgymrl/chemgymrl). If you wish to make a lot of changes to the library and implement custom reactions, extractions etc. we recommend that you simply work out of the repository rather than install it as a library.
 
```
pip install "git+https://github.com/chemgymrl/chemgymrl.git@main"
```


## Demo

If you want to demo the benches, you will need to install the gymnasium classic control dependency. 
```
pip install gymnasium[classic-control]
```


Then you can play the benches with the following command:
```
python -m chemistrylab.demo
```

The program allows you to select 
- [Extraction Bench Demo](https://docs.chemgymrl.com/en/latest/chemistrylab.benches.html#chemistrylab.benches.extract_bench.WurtzExtractDemo_v0)
- the [Reaction Bench Demo](https://docs.chemgymrl.com/en/latest/chemistrylab.benches.html#chemistrylab.benches.reaction_bench.FictReactDemo_v0)
- or the [Distillation Bench Demo](https://docs.chemgymrl.com/en/latest/chemistrylab.benches.html#chemistrylab.benches.distillation_bench.WurtzDistillDemo_v0)

