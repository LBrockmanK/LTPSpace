# chengImaging

Folders

Models: Stores previousely trained models
	File names will be a shorthand for the model details that is TBD

RawInput
	Input files to be classified

Software
	Python files

TrainingData
	Prelabelled data for model training

Root Files
	models.csv : Record of previous training results storing model details for easy review, info will probably be redundant with filename shorthand
	train.bat to execute training

TODO:
	Determine model naming convention
	Develop model trainers
	Probably a text or csv file to configure model generation parameters for testing

Resources:
	Statistics
		https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd

Final work plan:
	Data reading: Read CSV files
	To start will will differentiate collagen and Epithelium since we don't have cancer data yet, or maybe get all the possible classes we'll see
	We still start with just the convolutional neural network
	So read data -> Pass to model -> Generate and test model, all like previous project (just different data and different input layer)
		Note on data, first column appears to be the wavelength and second column the actual data, so the second is the only used for classification
	If we have time do the encoder