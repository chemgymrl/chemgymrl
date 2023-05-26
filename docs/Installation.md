[chemgymrl.com](https://chemgymrl.com/)

## Installation

In this tutorial we will be going over how to install the chemgymrl library. This can be done by installing straight from our git repository. If you wish to make a lot of changes to the library and implement custom reactions, extractions etc. we recommend that you simply work out of the repository rather than install it as a library.
 
```commandline
pip install "git+https://github.com/chemgymrl/chemgymrl.git@rewrite"
```


### Demo

If you want to demo the benches, you will need to install the gymnasium classic control dependency. 
```
pip install gymnasium[classic-control]
```


Then you can play the benches with the following command:
```commandline
python -c "import gymnasium,chemistrylab;from gymnasium.utils.play import play;play(gymnasium.make('WurtzExtractDemo-v0'))"
```
In this demo, the controls are set to the number keys. There are 10 actions in total mapped to keys 1,2,3,4,5,6,7,8,9,0.