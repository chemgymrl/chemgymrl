# Meeting Notes

## Date: June 16, 2020

### Attendees
- Mitchell Shahen (Secretary)
- Chris Beeler
- Mark Crowley
- Isaac Tamblyn
- Sriram Ganapathi

### Overview of Topics and Discussion
- ODEWorld and ExtractWorld are to be renamed Reaction Bench and Extract Bench, respectively
- Wurz reactions
- status updates
- create simple demo for a few reactions for next week

### Updates on Open/On-Going Projects and Assignments
- Chris: 
    - explained Wurz reaction, see below
- Mitchell:
    - take on the software task of implementing Wurz reactions in `Reaction Bench`
    - software setup and refactoring
    - [ ] will meet with Chris this week to start working on that
- Nouha:
    - away, sick 
- Sriram:
    - worked with Chris on Wurz reaction too
    - made a new branch to reformulate some of the RL relevant code in `Extraction World`, for the `Vessel` and `Material` classes
    - ODE World now has proper `Vessel` instances


###  Wurtz reactions
- They used to be heavily used for making polymers. Not used as much these days because of side-effects, but it has the right properties for everything we are talking about
- reaction properties
    - 2 R-X + 2 Na --> R-R + 2 NaX (in general)
    - hydrocarbons combine with sodium
    - good polarity properties (non-polar result and polar by-products)
    - creates butane chains, so fuel?
    - one step reaction, from the chemist's point of view
- can use any hydrocarbon
    - X : any halogen instead of chlorine
    - always use sodium
    - R : hydrocarbon chain, the desired result
- measures
    - A particular length of hydrocarbon chain could be a the desired result.
- solvents
    - diethyl ether (a "dry" liquid) - very non polar
    - sodium is very polar
    - sodium also very reactive with water
    - vessel must be closed to prevent contact between sodium and water in the air
- side reactions
    - Using large chloroalkanes can create alkenes - we could see these are impurities
    - Can limit the size of reactant chloroalkanes to prevent unwanted alkenes
    - A similar reaction produces asymmetric hydrocarbons, which can be considered unwanted, but allowed by-products
        - building the incorrect hydrocarbon chain can be detected and penalized 
- Reasons for Interest
    - Fits well with the uses and abilities of the Reaction and Extract benches
    - Well-known and well-researched process
    - Only one step from the chemist's point of view
    - Uses a non-polar solvent.
- Software Considerations
    - there should be a reaction class that can:
        - check that the materials are applicable to this bench 
        - does the time of reaction happen to equilibrium or not?

### Future Work
- Chris: help Mitchell with Wurz design
- Mitchell: software implementation of wurz reactions
- Nouha: preparing to present PPO, her own code
- Sriram: 
- new benches?:
    - Heating Bench for evaporating materials
    - Combustion was mentioned to be included in the Reaction bench
        - add formula for combustion reaction and use Reaction Bench

### Ongoing Discussing RL Papers
- [X] DQN was discussed briefly (Mitchell)
- [X] TD3 was discussed briefly (Sriram)
- [ ] PPO was intended to be discussed (Nouha), next week
- [ ] Further DQN, PPO, and TD# discussion was pushed to the latter half of the meeting next week.
