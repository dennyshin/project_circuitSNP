Denny Shin

circuitSNP project

Masters Research Project
Supervisor: Heejung Shim

Python Environment:
Use the circuitSNP conda environment stored in C:\Anaconda3\envs\circuitSNP

	circuitSNP environment:
		Python3.7.3
		- numpy 1.16.0
		  conda install numpy==1.16.0
		
		- matplotlib 3.0.2
		  conda install matplotlib==3.0.2
		  
		- sklearn 0.20.2
		  conda install scikit-learn==0.20.2
		
		- pytorch 1.1 (torch 1.1.0, torchvision 0.3.0)
		  conda install pytorch==1.1.0 torchvision==0.3.0 cpuonly -c pytorch
		

torchmodel1.py
    - my first working model
	- this was trained for only one tissue type
	- need to streamline the data preprocessing steps
		- are there any regions that we may want to filter out?
		- re-evaluate our choice of regions
	    - make use of pytorch's Dataset, DataLoader modules
			
model2_allmotifs.py
	- fixed the NN architecture to actually have 2 hidden layers
	- has 301bp long regions
		- this is incorrect and fixed in next version
	- initialise weights with xavier uniform for all layers
		- emprically shows quicker learning at start
		
model3.py
	- all models now work with all motifs from here on
		- merely change the dir of the motifs if you want to work with a subsample
	- has correct 300bp long regions
	
model3_longer.py
	- 100,000 epochs instead of 1000
	- 1000 epochs seems to converge at around 0.23 loss
	- 1000 epochs took around 3 hrs +- 20 mins

need to start implementing dsQTL and centiSNP data

not all files are the same as the ones on spartan