
# Meeting Notes

## Date: October 28, 2020

### Attendees

- Mitchell Shahen (Secretary)
- Isaac Tamblyn
- Sriram Ganapathi
- Nouha Chatti
- Chris Beeler
- Mark Crowley

### Updates

- Sriram:
    - Started running experiments and implemented some reinforcement learning
    - Found some issues or inconsistencies in chemistrygym (reward function, documentation and README files, extract bench has a discrete action space)
    - Issues were created in Github for some of the above

### Further Work to be Done

- Need use cases
- Clean up the code such that it is never unclear where the workflow comes from
- Need to specify what the repositories are used for and make this easy to translate into the final paper
- Reward function needs to be ironed out
    - Need a single reward function
    - This function is reference whenever a reward is needed
    - This function needs to be compatible for use with all benches
    - This function needs to have a rewards cap (no infinite rewards)
    - Chris, Sriram, and Mitchell will look at the code base and fix or implement these changes
- Determine if the lab manager is to use a single algorithm or multiple algorithms for benches to use
    - This is dependent on the action space of each bench
    - Extract bench employs a (large) discrete action space rather than a continuous action space
    - Explore if the extract bench action space could be changed to a continuous action space
- The state is very large (and therefore expensive)
    - Explore the implementation of an observation function to show what information is to be made available to the agent
    - Add a mass spec bench that gives a view of the state
- Succinct diagrams are needed
    - Visual representations of the benches, vessels, and their capabilities
    - Workflow is visible
    - Intuitive method to demonstrate what is happening and what can be done at each stage
    - Direct translation into the final paper
