Just a brief rundown of everything in the folder:

controllers - contains the final controllers that load in the solved agents, along with the other controller files that were used for training and 		experimenting in the different environments

SP#_Agents - contain either fully trained or partially trained agents, the best agents that were created for each environment are already in the controller 		file for the environment in ther BEST_AGENT subfolder

worlds - contains the environments that were made to train in

SOLVED_AGENT_VIDEOS - contains recordings of the simulations after the agents had trained on and solved their environment

INSTRUCTIONS:
To properly set up python environment to use webots and the created RL controller files:
1) Install Anaconda3, Webots (2023 is what I used) and launch Anaconda Prompt

2) create new env like: conda create --name webots_RL python=3.10

3) 'pip install' each of the following items:

	- Torch (pytorch)
	- webots
	- deepbots
	- tensorboard

4) enter 'where python'

5) copy python.exe path that is in your newly made conda environment

6) Launch Webots

7) Go to Tools > Preferences > General

8) paste the python.exe path into the "Python Command:" field and hit OK

--> You should now be set up to use the world files and controllers in this directory
