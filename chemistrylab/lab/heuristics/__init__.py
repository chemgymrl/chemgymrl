import os,sys

#thisdir = os.path.dirname(__file__)

#sys.path.append(thisdir)

from chemistrylab.lab.heuristics.ReactionHeuristics import *

from chemistrylab.lab.heuristics.DistillHeuristics import *

from chemistrylab.lab.heuristics.ExtractHeuristics import *


WurtzExtractDemo_v0_args = dict(
    drain_act = 0,
    polar_act = 6,
    non_polar_act = 5,
    mix_act = 1,
    wait_act = 8,
    done_act = 9, 
    layer_pos = (0,100), 
    polar_color = 638, 
    non_polar_color = 518, 
    history=6,
    dest_layer_pos = (100,200)
)



GenWurtzExtract_v2_args = dict(
    drain_act = 1, 
    polar_act = 30, 
    non_polar_act = 25, 
    mix_act = 9, 
    wait_act = 39, 
    done_act = 40, 
    layer_pos = (0,100), 
    polar_color = 638, 
    non_polar_color = 518, 
    history=2,
    dest_layer_pos = (100,200),
)

WaterOilExtract_v0_args = dict(
    drain_act = 1, 
    polar_act = 30, 
    non_polar_act = 25, 
    mix_act = 9, 
    wait_act = 39, 
    done_act = 24, 
    layer_pos = (0,100), 
    polar_color = 729, 
    non_polar_color = 518, 
    history=2,
    dest_layer_pos = (100,200),
)

WurtzDistillDemo_v0_args = dict(
    heat_act = 2,
    b1_act = 3,
    b2_act = 4,
    done_act = 5, 
    layer_pos = ((0,100),(110,210),(220,320))
)

GenWurtzDistill_v2_args = dict(
    heat_act=9, 
    b1_act=19, 
    b2_act=29, 
    done_act=30, 
    layer_pos = ((0,100),(110,210),(220,320))
)