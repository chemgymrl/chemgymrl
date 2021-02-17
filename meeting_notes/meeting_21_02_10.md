# Meeting Notes:

## Date: Feb. 10, 2021

### Updates and Tasks Moving Forward

- Amanuel:
    - Showed the group the website, documentation is up and available
	- Logos are available, but not shown
	- Website is all up on GitHub
	- Instructed to make the website live on the internet
	- Needs a better name (than RL Chemistry) (ChemGym?) (OmegaChem)
	- Pool on the team's conversation for the domain

- Nicholas:
	- Finishing up the lectures and Jupyter notebooks in extraction bench (latter shown in meeting)
	- Use "we" instead of "I" in jupyter notebooks
	- Add more images
	- Look at continuous integration methods (make sure any addition of new code does not break the program)

- Mark:
	- Finished distillation bench demo lessons
	- Discussed if the error that occurs when no more material is in boil vessel should occur. The answer: No!, let the user know there is no more material (or check the plots), keep heating up the air and vessel
	- Incremental heat addition is too high (increment heat many times per action) (use float instead of int)
	- Improve beaker and vessel naming
	- Plotting issues/bugs (moving materials between vessels)
	- Nouha suggested working on jupyter notebook and upload them and documentation to the website
	- Isaac suggests using more pictures
	- Add an is_broken flag for the vessel itself (add a critical, negative reward)
	- Fix decimal places (1 decimal place on the reward in the interim)

- Mitchell:
	- Lead the discussion about the log file, save log files in memory during an episode, allow the user to acquire certain elements in the log file, specify the precision
		- Log Files should be split into 2 parts: history (saves everything such that we can stop and experiment and use the history to restart it again from that point, used to make movies of multiple episodes) and lab notebook (what a scientist writes down in his notebook beside his desk (does not have access to the internal software of the system))
	- Merge the mshahen-2 branch into master

- Chris:
	- Look into unit tests (look into a thing called Travis?)
	- Look to add very simple reactions (not even useful RL reactions, just for introductions)

- Nouha:
	- Working on the PPO implementation
	- Getting results, but trying to get better rewards (playing with the hyper-parameters)
	- Do a one-on-one with Sriram