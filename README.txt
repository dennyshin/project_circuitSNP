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
	
model3_10ke.py
	- 10,000 epochs instead of 1000
	- 1000 epochs seems to converge at around 0.23 loss
	- 1000 epochs took around 3 hrs +- 20 mins

model3_100ke.py
	- 100,000 epochs
	- 100,000 epochs timed out with a 4 day limit

model4.py
	- out of 1,851,152 instances 70% training, 20% validation, 10% test
	- datasets were shuffled before split
	- 3 versions with 1k, 10k, 100k epochs exist
	- plotting included
	- lowered the learning rate to 0.0001
		- should use an adaptive learning rate: faster at start, slower later

model5.py
	- current architecture seems to converge at ~0.25 loss
	- should be set to ~5k epochs
	- no adaptive learning or early stopiing yet
	- includes confusion matrix for test predictions
	
model6.py
	- major directory changes
		- as such previous model scripts will likely not function properly
	- split the data prep process to a separate script
	- precision recall added
	- ROC added
	- area under PR, ROC curve added

model7.py
	- implement multiple models
	- 5k epochs
	- MLP
	- A: 5 5
	- B: 5 5 5 5 5
	- C: 5 5 5 5 5 5 5 5 5 5
	- D: 32 32
	- E: 64 64
	- F: 256 256
	- G: 256 5
	- H: 5 256
	- I: 32 32 32 32 32

--------------------------------------------------------------------------------------------------------------------

should use an adaptive learning rate: faster at start, slower later
need to start implementing dsQTL and centiSNP data

not all files are the same as the ones on spartan