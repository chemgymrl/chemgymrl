## Overview
- chris and mitchell met last week on distillation bench
## Details
- Mitchell not present - In the past week I have been working on the third distillation bench and intended to update the group on my progress with this new bench. I have created the engine and relevant vessel operations for the distillation bench and have moved on to plotting methods and testing the bench.

## Next Time
- live demos of each bench, even on whiteboard?
- major question: how do we communicate with each agent so it can know how to deal with multiple reactions/exrtactions etc
- each agent needst o see the relevant state information
- simple single slide for each bench
    -  

## Benches
- extraction bench
    - currently training targetted towards polar, nonpolar extractors
    - extracting salt would work for any salt
    - extracting butane would be different
    - input: 1x vessel
    - needed: we need an observation state to tell us which it is
    - output: 3x vessels output (desription , description )
    - reward: uses the target material to learn how to make it
    - reward questions:
        - should the output  
        - can we hardcode merging two vessels with simlarity purity? 
    - test out  
- reaction bench
    - currently training targetted towards a particular material
    - input: vessel
    - output: ?x vessels
    - reward: uses the target material to learn how to make it

## github
- everything is in the chemistrygym github repo now
- demo's too 

## Mass-Spec Bench
- [ ] make this
- output the peaks at the right place
- destroy the right proportion of material
- how to feed the output into an RL agent
    - return the list of pairs [(frequency, intensity)...(f,i)]

## Isaac wants more benches
    - Reaction Bench - 
        - UV-Viz Spectrum is one of the outputs

## Learning For the Lab Manager itself
- [ ] ongoing discussion 
- model based - where the 
- MCTS with aggressive tree pruning - 
- hierarchical RL
    - use a very constrained lab manager 

## Fundamental Question
- what is the interface to the benches?
- extraction
- reaction
- lab manager 
    - if the lab manager sends the target molecule 
      

         
## Update on UofT Material Design Robots
- Alán Aspuru-Guzik lab
