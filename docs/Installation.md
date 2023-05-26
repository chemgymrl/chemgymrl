[chemgymrl.com](https://chemgymrl.com/)

## Installation

In this tutorial we will be going over how to install the chemgymrl library. 



### Install Library
Now that everything is set up we simply seed to install the library. If you wish to make a lot of changes to the library and implement custom reactions, extractions etc. we recommend that you simply work out of the repository rather than install it as a library.
 
```commandline
pip install "git+https://github.com/chemgymrl/chemgymrl.git@rewrite"
```



### Demo

If you want to demo the benches, you will need to install the gymnasium classic control dependency. Then you can play the benches with the following commands:
```commandline
pip install gymnasium[classic-control]
python -c "import gymnasium,chemistrylab;from gymnasium.utils.play import play;play(gymnasium.make('WurtzExtractDemo-v0'))"
```