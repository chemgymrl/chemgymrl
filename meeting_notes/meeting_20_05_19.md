# Meeting Notes

## Date: May 19, 2020

### Attendees

- Chris Beeler
- Colin Bellinger
- Nouha Chatti (Secretary)
- Mark Crowley
- Sriram Ganapathi
- Mitchell Shahen 
- Isaac Tamblyn (Meeting Organizer)

### Topic covered:

* Updates about Subworld:

The agent isn't able to learn on the environment but there is no issue with the environment itself
Chris applied Ppo, A2C and DQN

Using the ppo alogirithm the agent was able to reach the goal but the policy isn't learning to achieve a higher reward.

Common issue with stables baselines: the policy could be stuck in a local minimum

* Possibility of training the agent based on a human experience?

Learning from demonstration vs inverse reinforcement learning: 
 
Learning from demonstration needs to record the actions taken, look at the states and evaluate them.

Inverse reinforcement learning: implement a reward function that gives a high reward to imitate the actions in the beginning and then update the policy to learn based on this experience. 
The goal wouldn't be to copy the trajectory but to learn from it instead.

Sriram suggested implementing rule based experts with simple if-then rules that could be used to generate trajectories.

* Playing the games with the keyboard

Playing the games with the keyboard for people to use it and
adding animations and sounds.

For subworld using the arrow keys, left and right would change the angles up/down would increase/decrease the throttle, another key to dive and surface and enter would execute the action


Subworld is set up to look at the current frame, the previous position and its dead reckoning.

* Discussion about using the dead reckoning on policy making. 
(Demonstration on white board was recorded)