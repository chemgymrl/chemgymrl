# Meeting Notes

## Date: July 21, 2020

### Attendees

- Mitchell Shahen (Secretary)
- Isaac Tamblyn
- Sriram Ganapathi
- Nouha Chatti
- Chris Beeler
- Mark Crowley

### Overview of Discussion

- Nouha reviewed and presented a promising curriculum learning algorithm
- Mitchell released demo videos using both benches

#### Review of Curriculum Learning Research

- The [paper](https://arxiv.org/pdf/1910.07224.pdf) reviews teacher-student curriculum learning techniques used in a BipedalWalker environment.
- POMDP is used in the approach outlined in this paper.
- The teacher selects new parameters that are mapped to a task distribution system.
- The teacher strives to maximize the student's final competence in completing the delegated tasks. Meanwhile, tasks delegated by the teacher increase in difficulty with each new task assigned.
- The students are "black boxes" to the teacher in that the teacher only sees the results of the tasks and not the inner-workings of the students. Additionally, all reinforcement learning is done by the student when assigned tasks, therefore the teacher agent is not privy to the reinforcement learning itself.
- Students "master" a task when they achieve a certain cumulative reward level.
- The paper outlined "Oracle", the algorithm when the agents are given, by an expert source, more information than they would normally have and achieve greater results. Two implementations used in the paper, RIAC and ALP-GMM, are promising as they both exceed the Oracle's scores numerous times.
- An implementation similar to RIAC lends itself well to the chemistrygym project due to similarities in the state variables.
- No RL algorithms have outright dominion in performing curriculum learning. Any demarcation is due to differences in the environment

#### ChemistryGym Demo Videos

- Two videos were released during this meeting of demos illustrating the compatability between the two existing benches (Reaction and Extract) using the Wurtz reactions.
- The first video, posted at 09:38, shows the React and Extract benches working with one of the six Wurtz reactions: 2 1-chlorohexane + 2 Na --> dodecane + 2 NaCl.
- The second video, posted shortly after at 09:39, shows the two benches reacting and extracting all six Wurtz reactions.
- With the results of the extract and react benches validated, work on the third bench, the Distillation Bench, will start.
