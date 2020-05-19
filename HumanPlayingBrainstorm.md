# Human Playing Brainstorm
**Task**: What are the setup for each game/bench for making it a human playable game including `actions`, `state observations`, `feed wards/points`, etc.

## New Tasks to implement
- [ ] Mitchell : add these features to git issue tracking as tasks to be assigned
- [ ] **Consecutive states:** see consecutive states when seeing result of action. Just like in Atari
    - How many 'frames' of action should be adjustable
    - By default we allow the agent to see a stream of all the states that intervened between their previous action and the current timestep. (eg. in Subworld this is six minutes of game time)
- [ ] **Zero'th order model**: 
    - In each domain we want to explore the impact of including zero'th order, or "common sense", models in the game. This like a *hint* to the agent that people have found can be very useful for making good decisions, or at least avoiding disasters.
    - Some examples:
        - Liquid Overflow (LO) : `(Total volume of destination vessel) - Sum_v (amount of liquid in vessel v)` for *ExtractionWorld*
        - Dead Reckoning (DR) :  `pos_{t+1} = (\tilde{x}_{t+1},\tilde{y}_{t+1})` estimate for next position of sub given current position (estimate or known form GPS) and expected movement distance at current throttle and no current for *SubWorld*.
        - **question:** can we actually provide this to the agent at decision time since it is dependent on the action they are about to take? We *could* provide the deadreckoning for the default action of "maintain course and throttle". Then we assume the agent will learn it's own DR model from experience.
        - ??? for *ODE Bench*.

## ODE World (Bench)
- actions
  - [ ] 
- states
  - [ ] normal states
  - [ ] zero'th order model?
- time
  - [ ] 


## Extraction Bench

- actions
  - [ ] 
- states
  - [ ] normal states
  - [ ] zero'th order model?
- time
  - [ ] 

## SubWorld
- actions
    - previously set heading
    - throttle/velocity
    - current position (**question:** do we maintain a seperate true position and estimated position? If we don't get GPS for a long time, do we maintain the last *known* position?)
    - last known position?
    - map of local area? (not always available? last observed map?)
- states
    - $$p_t$$ current position if GPS available
    - $$p_{t-1}$$ previous position
    - $$\tilde{p}_{t+1}$$ dead reckoning (sees *zero'th order models*) of next position *assuming* the agent *maintains* course and speed.
- time
  
    - every *six minutes* an action happens
    - what does the player/agent see of the states in-between? a video? all the states? just a summary abstraction?
    
    
