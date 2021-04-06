# Meeting Notes
## Date: November 25, 2020
- **Meeting Summary:** Focussed on implementation of reward model in current design of chemistry gym.
- **Present** : Isaac, Mark, Mitchell, Sriram

### Michell 

- looking for standard settings for constant settings of Wurz reaction
- also looked at reward function details

### Sriram
- He looked more deeply at the "human in the loop" paper
    - he looked at the code and setup
    - none of them do continous action space
- Still an open question 
    - what kind of setup we want to use
    - how do we highlight what are the challenges in this domain?
- The answer depends on:
    - the MDP structure
    - the reward function
    - the state, action space - discrete + continuous
    - dynamics definition
    - expensive observation actions
- Strategy
    - build a specific example of a domain like wurz reaction 

### New Resources to Look At 
- Isaac shared links to papers from a major robot chemistry lab in UK which he has spoken to
- they have very accessible papers and nice videos
- Isaac spoke to Leroy Cronin about using our algorithms, once they are ready, and the were excited about the prospect.
- We should all take a look at these papers to help get used to the type of setup and language used in this domain.
- Links:
    - https://science.sciencemag.org/content/363/6423/eaav2211/tab-figures-data
    - https://www.theverge.com/21317052/mobile-autonomous-robot-lab-assistant-research-speed
    - https://www.sciencedirect.com/science/article/pii/S2589597419301868
    - https://www.liverpool.ac.uk/cooper-group/

## Current Design
### States 
- each agent could have access to a timer of how long they've been in this bench

### Actions 
- Is there an active "done" action on each bench? there should be

### Reward Function 
- Reaction : 
    - encourages final compound being pure
    - encourgages all of final compound being in one vessel
    - `reward.py` file contains the ReactionReward class and other classes
        - functions can be called arbitrarily throughout the processing
        - rewards are computed at every step
        - reward function setup :  `reward = desired_material_amount/total_material_amount`
- Extraction : 
    - purity of each container (ratio of desired material in a vessel)
    - can also weight them to penalize ending up in mulitple vessels at the end `(total reward/number of vessels)`
- Distilation : 
    - multiple vessels
    - parameter is input heat, and the materials change, no time involved, heat added incrementally 
    - extracted material doesn't boil away, it goes into a second beaker
    - continue distilling until first beaker is pure


## Proposals : New Ideas to Consider Integrating into the Design 
- Extraction : 
    - material dependent purification cost
        - if extracted material in vessel is below desired purity level, then receive a low reward, and continue extracting

## Motivation: Why this is interesting 
- Interesting problems that don't come up in other domains
- eg. when one of multiple vessels has a high enough purity of the desired compound, then we may want to simply put it aside and focus on a smaller vessel. So the state, or resources can be divided up and reorganized in the middle of a task. This is a complex action that doesn't have obvious analogies in normal video games.
- can we come up with other examples?
