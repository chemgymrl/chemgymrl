[chemgymrl.com](https://chemgymrl.com/)

## Installation

In this tutorial we will be going over how to install the chemgymrl library. This can be done by installing straight from our git repository. If you wish to make a lot of changes to the library and implement custom reactions, extractions etc. we recommend that you simply work out of the repository rather than install it as a library. Note: This version was developed for python 3.8.
 
```commandline
pip install "git+https://github.com/chemgymrl/chemgymrl.git@main"
```


### Testing

To help verify that installation is working, you can try the following command:
```commandline
python -c "import gym,chemistrylab;env=gym.make('GenWurtzExtract-v2');env.reset();print(env.action_space)"
```
