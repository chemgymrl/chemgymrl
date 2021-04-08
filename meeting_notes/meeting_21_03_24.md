# Meeting Notes:

## Date: March 23, 2021

### Updates:

- Amanuel:
    - Working on the docs
    - Meeting with Mark B shortly to get more information for the website

- Nicholas:
    - Updates to lab manager and reaction bench
    - Got the new ODE solvers to work and timed them to find the optimal solution
    - Met last Friday and Monday to merge and clean up the Github
    - Updated the agent class to work with the lab env and benches
    - Made a demo video of the lab manager and walked us through the demo briefly
    - Currently the instructions for the lab manager is not available or documented
    - The lab manager is intended to see the input and output only currently
    - The lab manager action space includes the bench, the type of experiment, the vessel and then the agents
    - Lab manager has 2 modes (heuristic and human) similar to the render and human render
    - Examples or use cases of the lab manager would help demonstrate the full functionality of the lab manager code
    - The current ODE solver method is dependent on scipy (therefore the user must have access to scipy)
    - Read the docs is good to go

- Isaac:
    - Requested that no meeting notes be in chemgymrl
    - Changed the name of `master` branch to `main`
    - Ensure the Github has a minimal number of branches
    - We are clear to delete/archive chemistrygym (our private repo)

- Mark B:
    - Finished the plotting functions in reaction bench
    - Helped Nicholas fix a solute dict error
    - Transferred all the issues from the old repo to the new repo
    - Working on reaction bench compatibility Meeting with Amanuel tomorrow to go over summary
    - Recieved a medal of valour from Issac for his valiant efforts in moving over the issues from the old repo

- Nouha:
    - Implementing PPO on the reaction and extraction bench
    - Adding cost to the reward function (with purity)
    - Will post an overview of the cost function so everyone has access to it and can provide input
    - Looking to implement algorithms on the distillation bench
    - Looking to implement more RL algorithms throughout

To-Do:
- Need to have the GPL3 licensing in all of our code because we are public now.
- Need to have a definite licensing agreement by next meeting (@IsaacTamblyn and @MarkCrowley).

Necessary for Ship-able Product for Apr. 7:
- Nicholas and Mitchell have to consolidate the reaction bench ode issues
- Need the lab manager finished which requires the analysis bench to be created
- Unittesting module to be made in parallel with
- Documentation wrap-up
- Update install.py
- Run base algorithms and show that our environment supports RL
