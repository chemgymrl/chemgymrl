# Meeting Notes

## Date: June 23, 2020

### Attendees

- Mitchell Shahen (Secretary)
- Chris Beeler
- Nouha Chatti
- Isaac Tamblyn
- Sriram Ganapathi

### Overview of Topics and Discussion

- Attendees discussed their work in the past week
- Discussed Reaction and Extraction benches
- Discussed ways to integrate RL into chemistrygym

### Updates on Open/On-Going Projects and Assignments

- Chris, Sriram, and Mitchell:
  - Worked on merging Reaction bench and Extract bench.
  - Reaction bench performs 6 Wurtz reactions and outputs a vessel.
  - The vessel is used in Extract bench to separate materials.
  - Literary research of RL algorithms.

- Nouha:
  - Continuing research of Proximal Policy Optimization (PPO).

### Future Work

- Distillation bench should wait until Reaction and Extract benches are working completely. An introductory Distillation bench might be mixing salt and water then heating off the water.
- Sriram should work on outlining a paper for this project. This will help us frame the project moving forward. Aims to educate the user-base about what is and is not possible in our domain. Plots from Reaction bench are also useful to this paper.
- Implementing RL and how this is done is to be solidified.
  - Which algorithms will work best in our domain?
  - Which algorithms can be realistically implemented?
  - Does using human-in-the-loop make sense for this project?
  - Is including principles of chemistry feasible?
- Chemistrygym has important characteristics:
  - Vessel implementation
  - Benches acting as functions
  - A hierarchal structure of agents (lab manager, bench agents)
- Good approach is to combine tree-search and options methods.
- Useful to look at the [BabyAI](https://arxiv.org/pdf/1810.08272.pdf) paper.
- The project needs a RL hook to stand out.
- Partial observation may be implemented later.
- The hierarchal structure of the agents looks like the most interesting approach.
  - The lab manager overseas all benches and can assign agents to benches based on cost and required purity of products.
  - Many different types agents can be delegated to a bench.
  - Bench agents differ in their cost and purity of products.
  - I.e. high school students have low cost and graduate students have high cost.
  - Students can "graduate" to higher levels based on how many episodes of experience they have. I.e. HS --> US --> GS.
