# Proj Deep Chem RL
## Meeting Sep 30, 2020
### Compute
- computing instructions are in the mail and on the wiki Issac has sent
- V100s with 32GB per card, 96 Gb memory, 120 nodes (or cores)
- uses slurm

#### Goals for Problems: 
- open access RL problem
    - relevant to RL, good for society
    - simple algorithmic examples of using this domain
    - make it usable by everyone and start a community
    - providing a harder problem that come up in the real world
        - we need to show it isn't working for the baselines everyone likes
    - may need to also use ways that work for montezuma's revenge for example 

### Options for Going Forward
- level 1 - applying TD3, PPO or SA-Critic and other standard RL algorithms that work on continuous action spaces
- level 2  - apply existing curriculum learning algorithms on 
- level 3 - find a way to make curriculum learning for continuous spaces
- Other important issues to explore *at some point*:
    - biasing towards 


### Experimental Pipeline
- sriram: set up experimental pipeline for TD3
    - what are the experiments and domains?
    - try TD3 or Soft Actor Critic on simple two bench tasks
- tasks to learn that require two benches:
    - should work any wurz reactions - use dodacane
    - eg. dodecane salt mixture in an oil -> extract the salt the
    - what is the proper reward function? 

#### Issues
paper 3 - would also need to use other continous control domains for curriculum

### Topic Focus
#### Model Based RL Approaches
- What is new in this area in RL?
- build a prior model of the system and use that to learn
#### Curriculum Learning
- how to represent action space?

#### Policy Gradients using the Teaching Dimensions Approach?

### Status
#### Nouha
- very busy with three courses

#### Chris Beeler
- also busy with two courses

#### Sriram
- read the two papers `\cite{gottipati2020learning,hui2020babyai}`
    - they are most relevant but not that relevant for us 
- bit of a survey on curriculum learning elsewhere
- **Problem**:
    - No one has done continuous based learning for curriculum learning

#### Mitchell
- busy with 5 courses
- read over the two papers
