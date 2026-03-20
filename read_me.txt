-folder:
	- square_env: files for square_environment
	- box_env: codes for box_env

-files:
Both folders have same named files, the codes are different though.
	- first_train.py: Training of model on first trajectory
	- retrain2.py: Further training of value model
	- networks: contains a class with the actor and critic networks.
	- plot_value.py: file used for all the plotting
	- real_vis_traj.py: file used for real time plotting of anything
	- GEN.py: main file to generate GEN traj output
	- GEN_func.py: functions for GEN.py
	- corr_xy.py: function to map LAHN output to the position.
	- actor: all the files starting with actor are the actor network files
	- DDP_thesis: The final thesis for the project

- remarks: 
	-The folders do not include the data required to generate results. 
	-The folders do no include the files for VDON model.
	-The folders do not include files to generate trajectory.
	-All the above are not included because of size contraints because the dataset sizes were nearly 1GB
	-reason for not adding the VDON and trajectory codes is because they are not written by me 
