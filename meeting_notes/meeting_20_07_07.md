# Meeting Notes

## Date: July 7, 2020

### Attendees

- Chris Beeler
- Nouha Chatti (Secretary)
- Mark Crowley
- Sriram Ganapathi
- Mitchell Shahen 
- Isaac Tamblyn 

### Topic covered:

Chris:

* Used Relu as activation function for his previous tests which seems to be not be the best activation function to use.

* Changed RELU to Tanh in every layer, and as a result the agent was able to learn up to the level that has islands.
	
* Discussed Activation functions, and the impact of  ReLu (including the fact that it throws away the negative feedback)

* Tanh works well for the chart network.
	
* Ran other tests using Exponential linear unit, and obtained similar results as it performs well in the start but decreases as the environment gets more complex.
	
* Testing of the removal of level 4 and 5 because comparing to the first 3 level, the chart network inputs are not constants anymore which makes it more difficult.
	
* Added the secondary levels that transition between the levels.  

* Applied TD3 

Isaac suggested that instead of ending the episode, adding a large negative reward if the agent hits the island and then increasing this penalty. 
Mark mentioned that ramping-up the reward on a specific step approach for the curriculum learning could be helpful.  

Nouha:

* Presented policy gradient methods and the ppo algorithm

* Trained the gfootball environment to implement ppo
	
Mark and Isaac recommended recording a video of the game/environment
	

Mitchell: 

* Working with chemstrygym using the reaction that Chris presented before.
	
* Working on the Reaction bench and extract bench to work and improving the interoperability of Chemstrygym to make it compatible with other reactions.
	
Isaac suggested adding the distillation Bench 

Sriram: 
* Looked through curriculum learning Algorithms to find a new one to implement
* It seems not clear how to do curriculum learning with continuous action spaces. 

When using discrete actions, the problem won't be practical if there's many action, because they are usually represented as continuous in this case but then with clustering techniques it is possible to go from continuous actions to discrete when executing the actions (Synthesis paper is an example) 
	
Future work: 

* Mark suggested looking through policy gradient methods to do curriculum learning in continuous space to compare it to discretizing the action space. 

* For extract world there's 2 actions : 1 pick the process and then the duration, Chris mentioned that it could be discretized since already one is discrete.  But for the distillation bench if it's discretized then there could be some loss. As for the react bench it could be discretized by choosing increments approach so instead of adding everything, adding percentages would break the continuous actions.  