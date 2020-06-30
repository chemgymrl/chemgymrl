# Meeting Notes

## Date: June 30, 2020

### Attendees

- Mitchell Shahen (Secretary)
- Isaac Tamblyn
- Sriram Ganapathi
- Kyle Sprague

### Overview of Topics and Discussion

- Sriram reviewed various RL papers
- The benefits and feasibility of various RL algorithm implementations were discussed

### Reinforcement Learning Research Discussion

- Curriculum Learning encompassed many different types of learning strategies
- Skill-Based Learning:
  - Learn reusable and transferable skills
  - Values are assigned to learned skills indicating their efficiency and usefulness
- Curriculum Learning Through Distillation:
  - Create a set of networks
  - Learn as you progress by creating new networks in new layers
  - Child networks represent new learned skills
- Teacher Guided Learning:
  - A teacher agent assigns tasks to student agents
  - The assigned tasks start small, but progress to larger, more complex tasks
  - If a student fails to provide a sufficient result, the student regresses down to a smaller task
- Curriculum Learning Through Self-Play:
  - Similar to Teacher-Student Learning
  - Two Agents of the same level compete against each other and progress to take on more difficult tasks

- Hierarchal Learning in Minecraft:
  - Develops lifelong, reusable skills
  - Lifelong learning combats catastrophic forgetting
  - Catastrophic forgetting pertains to the loss of learned skills, which can severely hinder an agents ability to perform tasks
  - Catastrophic learning is more impactful in Minecraft as opposed to the repetitve nature of Atari-style games
  - Actions have a hierarchal perspective (options rather than strict actions)
  - Hierarchal Learning shows much better results than DQN in this environment

- Deep Reinforcement Learning with Monte Carlo Trees:
  - Learning through MCTS
  - MCTS looks into the future by jumping ahead, looking at the next state, and determining the optimal method for solving the next state with existing policies
  - MCTS is useful for our purposes as rolling forward allows the lab manager to delegate tasks, while predicting future states

- Using MCTS:
  - Using a discrete action space, the lab manager assigns tasks to bench agents
  - Only Bench agents learn, but their learning and the results of their learning is passed back to the lab manager
  - Meanwhile, the lab manager is concerned with rolling forward to future scenarios and delegating tasks
  - The lab manager recieves the results/scores/efficiency of the bench agents and can model the delegation of tasks based on this information
  - The lab manager -> bench agent relationship is similar to the teacher -> student relationship
  - The lab manager does not teach the bench agents, just creates the curriculum and assigns tasks (all learning is done by the bench agents in performing tasks)
  - Agents are trained and copies of policies are stored to be later loaded and used to build networks
  - Minecraft is a good starting point, however Minecraft has no lab manager -> bench agent system, but the algorithm is independent to this structure

### Future Work

- Estimations indicate it will take 2-3 weeks to implement, train, and test a MCTS algorithm for 10,000 - 20,000 episodes.
- The RL paper will mostly focus on ChemistryGym, but allow for the exploration into different disciplines and applications
- SUBworld will not be released with ChemistryGym (will be its own separate project)
- Theoretical Analysis on Curriculum Learning can be done to show relevant theory and prove the algorithms and implementation is rational
- Chris and Mitchell will adapt the algorithm for ChemistryGym, whil Sriram and Nouha work on the theory (an implementation of the algorithm in Minecraft will help demonstrate how the algorithm works)
