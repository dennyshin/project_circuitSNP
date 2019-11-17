# Project circuitSNP

Project circuitSNP is a neural network framework that implements both sequence level data and transcription factor motif footprints to predict the chromatin accessibility in tissue specific contexts.

Project circuitSNP was made as part of an ongoing research project. While the product is not fully ready to be used as a python package (as such, no bin files are provided), the scripts and functions are still able to be used in the user’s research contexts and goals.

## Data pre-processing

The steps to preprocess data have been automated for ease of use. The only things that may need to be specified are the directories for the data files containing the tissue specific footprints and also the path to the files containing the motif specific footprints. These are provided by default using LCL tissue but to the directory would need to be changed in order to specificy a different tissue. Simply running "dataprep.py" without any modifications will generate the appropriate neural network instances and would allow for downstream analysis of LCL tissue chromain accessibility.

#### dataprep.py

This script's main function is to generate the training instances and also split the dataset into three parts: train set, validation set, test set. By default each set contains 70%, 20% and 10% of the original dataset respectively. This can modified by altering the train_len and val_len variables. These indicate the ending elements of the original dataset. The test length is simply what is left.

The import_tissue_regions function allows entire tissue specific transcription factor binding site data files to be parsed into the circuitSNP framework. This function is specific to the data files that are found [here](http://genome.grid.wayne.edu/centisnps/).

The create_regions function cross references two tissue type data in order to create the relevan regions needed for instance generation. Simultaneously, it is able to label which regions have open or closed chromatin (again, specific for the tissue type inputted).

The import_motifs function requires a path to a directory which contains the motif specific transcription factor footprint data files. The function cross references the footprints found against the previously generated regions made using create_regions. This generates the binary instance features. Additionally, it saves the features' motifs such that they can be retrieved later as the column labels of each instance.

As long as the above functions' inputs are correctly given, dataprep.py is able to generate the input instances and output class labels. It saves all datasets, the regions and the motif column labels. By default the naming is part of the LCL analysis but this can also be modified.

## Model training and evaluation

Once the appropriate instances have been created, model.py can load and start to build the neural network. The model's architecture can be defined inside the class Net. All of hyper-parameters such as the learning rate, optimizer function, loss function and number of parameters can also be modified. Once the training loop occurs, it will complete until all requested epochs have occured. During the training loop the script continuously save the model parameters whenever a new minimum validation loss was found. After the end of the loop, the script will save the model's training progression in a plot and .txt files.

After model training, a separate script handles the evaluation. Using the test dataset, it calculates numerous model performance metrics as well as generating model plots.

#### model.py

model.py is set to load four data files: Xtrain, Xval, Ytrain and Yval. These correspond to the data files that would have been created in dataprep.py. If any file names were changed in dataprep.py the same would have to be altered here in model.py also.

pytorch allows the use of gpus but the this project has not integrated gpu support yet. As such, the default device is set to cpu.

Inside the Net class, the model's architecture is fully customisable. Fair warning, altering the input and/or output layers may cause the script to crash through dimensonality mismatches. This is at the deiscretion of the user. Any available pytorch functions and layers as well as custom python functions are able to be integrated also.

The model's default parameters are:
- learning rate = 0.0001
- Adam optimizer
- binary cross entropy loss
- 5000 epochs

The resultant plot and .txt files' directories may be customised to suit your needs.

#### predict.py

predict.py loads the Xtest and Ytest.

The chosen model to use should be defined again in the Net class. Once the class has been defined, the model parameters can be loaded as long as the .pt or .pth file's directory is given.

The following metrics are calculated: precision, recall, specificity and FPR. These metrics are calculated at every probability threshold (rounded to 2 decimal points). The resultant PRC, ROC, auPRC, auROC are also reported and saved. The directory destinations can again be altered if desired.

## dsQTL data preparation and analysis

With a fully trained and defined model, dsQTLprep.py and dsQTLeval.py can perform the variant effect analysis on a desired set of data. New instances would be generated and saved. Finally, the model's variant instance predictions will be saved and it will also report a "isdsQTL" indicating 1 if it is predicted to be and "0" otherwise. Note that if isdsQTL = 1, this means that the model has predicted that the SNP is a circuitSNP.

#### dsQTLprep.py



#### dsQTLeval.py



