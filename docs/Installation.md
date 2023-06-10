[chemgymrl.com](https://chemgymrl.com/)

# Installation

In this tutorial we will be going over how to install the chemgymrl library. This can be done by installing straight from our git repository. If you wish to make a lot of changes to the library and implement custom reactions, extractions etc. we recommend that you simply work out of the repository rather than install it as a library.
 
```
pip install "git+https://github.com/chemgymrl/chemgymrl.git@rewrite"
```


### Demo

If you want to demo the benches, you will need to install the gymnasium classic control dependency. 
```
pip install gymnasium[classic-control]
```


Then you can play the benches with the following command:
```
python -c "import gymnasium,chemistrylab;from gymnasium.utils.play import play;play(gymnasium.make('WurtzExtractDemo-v0'))"
```

```
Controls (Use the numpad or number row):
1: extraction_vessel performs Event(name='drain by pixel', parameter=[1], other_vessel=Beaker 1)
2: extraction_vessel performs Event(name='mix', parameter=array([-0.02]), other_vessel=None)
3: Beaker 1 performs Event(name='pour by volume', parameter=array([0.02]), other_vessel=extraction_vessel)
4: Beaker 2 performs Event(name='pour by volume', parameter=array([0.02]), other_vessel=extraction_vessel)
5: extraction_vessel performs Event(name='pour by volume', parameter=array([0.02]), other_vessel=Beaker 2)
6: C6H14 Vessel performs Event(name='pour by volume', parameter=array([0.01]), other_vessel=extraction_vessel)
7: diethyl ether Vessel performs Event(name='pour by volume', parameter=array([0.01]), other_vessel=extraction_vessel)
8: extraction_vessel performs Event(name='mix', parameter=[0.001], other_vessel=None)
9: extraction_vessel performs Event(name='mix', parameter=[0.016], other_vessel=None)
0: end experiment
```