# README

This repository contains the training data, the trained models and the python scripts of the publication **"Machine learning for ultra high throughput screening of organic solar cells: Solving
the needle in the hay stack problem."** in Advanced Energy Materials. 

To enable the reader to better reproduce or results, following data can be found:
* Directory /data containing the **training data** for the DCV-V-Fu-Ind-Fu-V:C60 model used in Fig.2 and the PM6-DTY6 model used in Fig.7 as csv-files
* Directory /trained_models containing the **trained ML-models** exported from tensorflow for the respective data in Fig.2 and Fig.7
* The **python environment** used to run the scripts for training and prediction as .yml file for use with the conda virtual-environment manager
* The **python scripts** to train the ML-model and predict lifetime and mobilities from experimental JV-curves 

Further, this readme file contains a **quick start guide** on how to use the python script along with more **detailed documentation**.

---

## Table of Contents
1. Pipeline overview
2. Quick start guide
3. Pipeline
   
   	1. Pipeline scripts
		* pipeline.py
		* train.py
		* predict.py
	2. Base classes
		* class\_nnmodel\_pipe.py
		* class\_ml\_data\_pipe.py
		* class\_ml\_evaluation\_pipe.py


4. Utility and plotting scripts
	* modules/opkm\_plots.py
 	* Set up logger path
	
---

### 1. Pipeline overview

This pipeline uses the python library keras tensorflow to train a Neural Network on a simulated training set of JV-curves from organic solar cell devices. The process of creating the training set with the drift diffusion model oghma\_nano is described in the main publication and in great detail in the oghma\_nano manual on https://www.oghma-nano.com/docs.html?page=Manual under Automation.
The main script **pipeline.py** calls the **train.py** script to train the ML-model on the training set and the **predict.py** script to use the trained model to predict the parameters from experimental JV-curves of solar cell devices.

---

### 2. Quick start guide

* To train the ML-model and predict on experimental JV-curves run the script **pipeline.py** in the repository
	* The pipeline takes a data set for training stored in directory /data in the form of e.g. training\_data\_DCV-V-Fu-Ind-Fu-VC60.csv
	* The experimental data to predict have to be stored in the base directory as a csv file (see training\_data\_pm6dty6\_example\_input\_exp.csv for reference, the current density in SI units has to be stored at the voltages simulated in the training set) 
	* The file containing all the units and names of the predicted parameters used for plotting are stored in a .json file (see training\_data\_pm6dty6plots.json as reference, use plotnames\_dict\_pipe.py to create such a file)
* Following variables can be set in pipeline.py:
	* 'dataset\_name': set name of data set file 
	* 'dataset_path': set path to the dataset
	* 'device\_path': set path to the experimental data file
	* 'devices': list of names of experimental data to loop over if needed
	* 'model_name': the name of the parameter to be predicted (column name of a target in the training dataset file)
	* 'model\_path': set path to save the trained model

---

### 3. Pipeline

#### 3.1 Pipeline scripts

##### pipeline.py
* The main script to train the ML-model and predict on experimental JV-curves
* Following variables can be set in pipeline.py:
	* 'dataset\_name': set name of data set file 
	* 'dataset_path': set path to the dataset
	* 'device\_path': set path to the experimental data file
	* 'devices': list of names of experimental data to loop over if needed
	* 'model_name': the name of the parameter to be predicted (column name of a target in the training dataset file)
	* 'model\_path': set path to save the trained model

##### training.py
	* Train the ML model on the training set
    	* Save best model
    	* Save plots of training progress
     
##### predict.py
	* Predicts a parameter using experimental JV curves
	* Saves the predictions in a csv file

#### 3.1 Base classes

##### class\_nnmodel\_pipe.py

* Base class for the neural network using keras tensorflow 
* https://www.kaggle.com/code/varun2145/how-to-iterate-over-keras-layer-as-a-parameter
* Create the neural network from input parameters
* Train (fit) the model to the training data
* Save the model if necessary

##### class\_ml\_data\_pipe.py

* Class machinelearning\_data\_struct
	* This class handles the splitting of the data set into features and targets, as well as the normalisation for the training data
* Class machine\_learning\_experimental\_data
	* This class handles the splitting of the data set into features and targets, as well as the normalisation for the esperimental data that we want to predict on (as they don't have any targets)
	* It takes the normalization data from the machinelearning\_data\_struct as a parameter to have the same normalization

##### class\_ml\_evaluatuation\_pipe.py

* The class handles calculation of error and plotting the training data and confusion matrices of the training and testing process

### 4. Utitlity and plotting scripts

##### model/opkm\_plots.py

* This script sets the figure style and plot settings for the pipeline

##### Set up logger path

* See module logging: https://realpython.com/python-logging/
* logging.config.fileConfig(fname=os.path.join(logger_path, 'file.conf')
* The logger path has to be set to the path where the 'file.conf' for the logger is set (example file in /logger)
* Logging level can be changed to DEBUG by logger.setLevel('INFO') to logger.setLevel('DEBUG')


