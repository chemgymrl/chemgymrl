# ChemGymRL

[![Build Status](https://travis-ci.com/chemgymrl/chemgymrl.svg?branch=main)](https://travis-ci.com/chemgymrl/chemgymrl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chemgymrl.readthedocs.io/en/latest/)

## Overview

[ChemGymRL](https://www.chemgymrl.com) is a chemistry laboratory environment populated with a collection of chemistry experiment sub-environments, based on the [Gymnasium](https://gymnasium.farama.org/) API for use with reinforcement learning algorithms.

It was created to train Reinforcement Learning agents to perform realistic operations in a virtual chemistry lab environment. Such operations are virtual variants of experiments and processes that would otherwise be performed in real-world chemistry labs and in industry. The environment supports the training of Reinforcement Learning agents by associating positive and negative rewards based on the procedure and outcomes of actions taken by the agents.

For more information, see the [ChemGymRL Documentation](https://docs.chemgymrl.com).

## Installation

Installing the chemgymrl library can be done by installing straight from our git repository. If you wish to make a lot of changes to the library and implement custom reactions, extractions etc. we recommend that you simply work out of the repository rather than install it as a library.
 
```
pip install "git+https://github.com/chemgymrl/chemgymrl.git@main"
```


### Demo

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
