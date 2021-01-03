# Reaction Bench
Modules defining reactions available to the reaction bench

## Environments

- get_reactions : A module to provide methods to assist in performing reactions, unpacking material classes, and accessing all available reactions.
- reaction_0 : Performs a sample reaction involving the combination of two reactants into a single product.
- reaction_1 : Performs a set of three sample reactions involving one parallel reaction and one reversable reaction.
- reaction_2 : Performs a set of four sample reactions involving several parallel reactions.
- reaction_3 : Performs a set of five sample reactions involving several parallel reactions.
- reaction_4 : Performs a set of three sample reactions involving two completely parallel reactions.
- reaction_5 : Performs a set of three sample reactions involving three parallel reactions using 5 different reactants and producing 4 different products.
- reaction_6 : Performs a set of five sample reactions involving multiple parallel reactions using 4 different reactants and producing 4 different products.
- wurtz_reaction : Performs a set of six Wurtz reactions, which are the reactions relating to the six different isomers of chlorohexane reacting with sodium to produce large chlorohydrocarbons.

## Overview

The reaction bench environment is intended to provide a set of available reactions to the reaction bench engine. Any one of the reaction modules listed above is designated by the reaction bench environment and passed to the reaction bench engine, both the environment and engine are in `../reaction_bench`. The reaction bench engine then performs the reactions contained in the reaction module.