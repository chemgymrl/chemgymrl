# Meeting Notes

## Date: June 9, 2020

### Attendees

- Mitchell Shahen (Secretary)
- Chris Beeler
- Nouha Chatti
- Mark Crowley
- Isaac Tamblyn
- Sriram Ganapathi
- Colin Bellinger

### Overview of Topics and Discussion

- Attendees went over their work in the past week
- Discussed RL algorithms at a glance
- Discussed future work in SubWorld
- Laid plans to combine tasks in CHEMWorld and ODEWorld

### Updates on Open/On-Going Projects and Assignments

- Chris:
  - Has had issues with PD3 and stable baselines in SUBworld.
  - SUBworld would spontaneously stop working after a couple hours.
  - Produced a curriculum learning version of SUBworld:
    - Implemented different difficulties
    - Difficulty parameters included start and end positions, water current, presence of islands.
    - Each difficulty parameter ranged from 0 -> 1.
    - Nothing has been trained on this implementation yet.
    - The motivation for adding difficulty incrementally was to try to train the agent to first manouvre, then navigate water currents, and finally avoid islands.

- Mitchell:
  - Has been researching Deep Q-Network (DQN).
  - Started working with algorithms in openai/baselines repo to boost understanding.

- Nouha:
  - Has been researching Proximal Policy Optimization (PPO).
  - Is working with Google Research's Football RL environment. The code runs, but the rewards are not sufficiently large, which is being resolved.

- Sriram:
  - Has been finishing up the deadline last Friday.

### Future Work

- Chris:
  - Investigate possible updates to SUBworld:
    - Investigate states and actions that are independent of the size of the area in which the sub can move.
    - Making the value function for each state available. Therefore, to avoid excessively large value functions, the resolution must be low.
    - Show the path that the sub has taken.
    - Display a heat map of the value/benefit of being in a certain location in the current state of the environment.
  - Work with Sriram and Mitchell to test the capabilities and usage of the CHEMWorld, ExtractWorld, and ODEworld modules.
  - Add elements from the CHEMworld module, such that they are usable in ODEworld.

- Mitchell:
  - Work with Chris and Sriram on testing the capabilities and usage of the CHEMworld, ExtractWorld, and ODEworld modules.
  - Investigate classes and reactions that need to be added in the long-term.

- Nouha:
  - Continue literary review and research into PPO and other RL algorithms.
  - Help in the development of algorithms.

- Sriram:
  - Work with Chris and Mitchell on testing the capabilities and usage of the CHEMworld, ExtractWorld, and ODEworld modules.
  - Alpha test the CHEMworld modules as if in preparation for writing a paper.
  - Run a bunch of reactions (7?) and see how they work.
  - Create a "presentation" of the reactions/procedures that do and do not work to assign future tasks. 
  - If using different __world modules together does not work, fixing this becomes the development priority.

### Discussing RL Papers

- DQN was discussed briefly (Mitchell)
- TD3 was discussed briefly (Sriram)
- PPO was intended to be discussed (Nouha), but was not due to time constraints
- Further DQN, PPO, and TD# discussion was pushed to the latter half of the meeting next week.
