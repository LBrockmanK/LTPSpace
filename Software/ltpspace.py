import math
import time
import random
from sklearn.model_selection import train_test_split, learning_curve
import copy
import numpy as np
import logging
import os
from datetime import datetime
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
import sys
import gc
import glob
import re
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import re
import glob
from sklearn.metrics import classification_report

# Parameters
class Parameter:
		"""Stores the attributes of a model parameter, as well as any functions needed to work with those parameters"""

		def __init__(self, name, min_value, max_value, is_integer=True, resolution=1):
			"""Initializes a parameter object with all of its rquires attributes"""
			self.name = name
			self.min_value = min_value
			self.max_value = max_value
			self.is_integer = is_integer
			self.resolution = resolution
			self.direction = True  # True = Up, False = Down
			try:
				self.stages = max(int(math.log2((max_value - min_value) / resolution)) - 1, 0)
			except Exception as e:
				self.stages = -1

		def __str__(self):
			"""Prints the attributes of the parameter"""
			return f"{self.name}: min={self.min_value}, max={self.max_value}, is_integer={self.is_integer}, resolution={self.resolution}, stages={self.stages}"

		def round_to_resolution(self, value):
			"""Rounds a number to the resolution of the parameter"""
			return round(value / self.resolution) * self.resolution

		def adjust_value(self, value, change):
			"""Applies a value change to the parameter, keeping it within the limits and resolution given by the attributes. Change input only has its magnitude considered"""
			# Check that any change will actually be made
			if(change == 0):
				return value

			# Ensure change has proper sign
			if(change < 0):
				change = -change

			# Ensure minimum magnitude of change is self.resolution
			if(change < self.resolution):
				change = self.resolution

			# Apply sign based on direction
			if(self.direction == False):
				change = -change

			# Apply change, still ensuring we keep to resolution and limits
			new_value = self.round_to_resolution(value + change)
			return min(max(new_value, self.min_value), self.max_value)

# Less Than P Space
class Explorer:
	"""This class performs hyperparameter tuning on an sklearn model via binary search hill climbing of the hyperparameter space"""
	def __init__(self, model, sample_data, sample_class, param_count, validation_split = 0.2, test_split = 0.25, debuglevel = logging.INFO, log_directory = ".", recover = False):
		"""Configures all required data for explorer to occur"""
		#From Inputs
		self.model = model # provide the function that will be used for scoring
		self.test_split = test_split # proportion of sample data (after removal of validation data) that will be used for testing during training
		self.param_count = param_count # Number of parameters to be used
		self.recovered = recover # Are we recovering from an interrupted run
		self.sample_data, self.validation_data, self.sample_class, self.validation_class = train_test_split(sample_data, sample_class, test_size=validation_split, random_state=42)

		#Other
		self.parameters = []
		self.stages = 0
		self.params = []
		self.maxrefinements = None
		self.maxtime = None
		self.iteration = 1
		self.score = 0
		self.starttime = 0
		self.currentparam = 0
		#TODO: Once complete, see what here doesn't actually need to be a class variable and can go into particular methods

		#Model details
		self.kfolds = 10 # Number of folds used for kfold cross validation
		self.metric = 'accuracy' # Metric used when compiling models
		self.lossf = "categorical_crossentropy" # loss function used when compling model
		self.scoring = accuracy_score # Scoring function used for evaluating model performance
		self.epochs = 10 # Epochs for fitting models
		self.batch_size = 32 # Number of samples to process before updating model

		#Set up logging
		self.log_directory = log_directory
		if not os.path.exists(self.log_directory):
			os.makedirs(self.log_directory)

		now = datetime.now()
		date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
		self.log_filename = f"ltps_{date_time}.log"
		logpath = os.path.join(self.log_directory, self.log_filename)
		logging.basicConfig(filename=logpath, level=debuglevel, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
		console_handler = logging.StreamHandler()
		console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
		logging.getLogger().addHandler(console_handler)

		# Disable all logging from matplotlib
		mpl_logger = logging.getLogger('matplotlib')
		mpl_logger.setLevel(logging.CRITICAL)

		# Disable all logging from PIL or Pillow
		pil_logger = logging.getLogger('PIL')
		pil_logger.setLevel(logging.CRITICAL)

	def modify_Model(self, kfolds = None, metric = None, lossf = None, scoring = None, epochs = None, batch_size = None):
		"""Override various parameters used in the training of models"""
		if(kfolds != None):
			self.kfolds = kfolds
		if(metric != None):
			self.metric = metric
		if(lossf != None):
			self.lossf = lossf
		if(scoring != None):
			self.scoring = scoring
		if(epochs != None):
			self.epochs = epochs
		if(batch_size != None):
			self.batch_size = batch_size

	# Add a parameter to the parameter set
	def addParam(self, min_value, max_value, name = None, is_integer = True, resolution = 1):
		"""Defines a new parameter for the optimizer"""
		# Check if we are adding more parameters than we have room for
		if(len(self.parameters) >= self.param_count):
			logging.info(f"Error, Too many parameters")
			return

		# Add new param
		new_param = Parameter(name, min_value, max_value, is_integer, resolution)
		self.parameters.append(new_param)
		logging.info(f"Added parameter: {new_param}")

		# Update stages
		if(new_param.stages > self.stages):
			self.stages = new_param.stages

		return

	# Check that all conditions are met to begin search
	def invariant(self):
		"""Check to ensure inputs are valid to begin optimization process"""
		errors_found = False

		if len(self.parameters) == 0:
			logging.error("No Parameters Defined, Aborting...")
			errors_found = True
		if self.param_count < len(self.parameters):
			logging.error("Missing parameters, aborting, Aborting...")
			errors_found = True
		if self.param_count > len(self.parameters):
			logging.error("Too many parameters, aborting, Aborting...")
			errors_found = True
		if self.model is None:
			logging.error("Model is not defined, Aborting...")
			errors_found = True
		if self.sample_data is None or (isinstance(self.sample_data, (list, np.ndarray)) and len(self.sample_data) == 0):
			logging.error("Sample data is not defined or empty, Aborting...")
			errors_found = True
		if self.sample_class is None or (isinstance(self.sample_class, (list, np.ndarray)) and len(self.sample_class) == 0):
			logging.error("Sample class is not defined or empty, Aborting...")
			errors_found = True
		if self.test_split is None or self.test_split <= 0 or self.test_split >= 1:
			logging.error("Invalid test split, Aborting...")
			errors_found = True
		if self.param_count is None or self.param_count <= 0:
			logging.error("Invalid param_count value, Aborting...")
			errors_found = True

		return errors_found


	def check_special_exit_conditions(self):
		"""Check conditions that would result in early termination of search"""
		if self.maxrefinements is not None:
			if self.maxrefinements >= 0:
				logging.info(f"Max iterations reached, Quitting...")
				return True
		if self.maxtime is not None:
			if time.time() - self.starttime >= self.maxtime:
				logging.info(f"Max time reached, Quitting...")
				return True
		if self.score >= 1.0:
			logging.info(f"Max score reached, Quitting...")
			return True
		return False

	def cross_validate_model(self, params):
		"""Begin testing a new model iteration, logging time and progress"""
		# Record start time
		iterationtime = time.time()
		
		# Test model
		kfold = KFold(n_splits=self.kfolds, shuffle=True)
		scores = []

		for train_index, test_index in kfold.split(self.sample_data):
			X_train, X_test = self.sample_data[train_index], self.sample_data[test_index]
			y_train, y_test = self.sample_class[train_index], self.sample_class[test_index]

			m = self.model(params, X_train.shape[1:])
			score = self.train_and_evaluate_model(m, X_train, X_test, y_train, y_test)
			gc.collect()
			scores.append(score)

		newscore = np.mean(scores)

		# Report and update time and iteration
		iterationtime = time.time() - iterationtime
		self.iteration += 1
		logging.info(f"Iteration:{self.iteration}, Time:{iterationtime}, Score:{newscore}")

		return newscore

	def train_and_evaluate_model(self, m, X_train, X_test, y_train, y_test):
		"""Train and test a given model"""
		y_train = to_categorical(y_train)

		m.compile(loss=self.lossf, metrics=[self.metric])

		early_stopping = EarlyStopping(monitor='val_loss', patience=5)

		m.fit(X_train, y_train, verbose=0, epochs=self.epochs, callbacks=[early_stopping], validation_split=self.test_split, batch_size = self.batch_size)
		gc.collect()

		return self.test_model(m, X_test, y_test)

	def test_model(self, m, X_test, y_test):
		"""Take a model and set of test data and return the score for the model"""
		predictions = m.predict(X_test)
		predictions = np.argmax(predictions, axis=1)

		y_test = to_categorical(y_test)
		y_test = np.argmax(y_test, axis=1)
		
		logging.info(classification_report(y_test, predictions))
		return self.scoring(y_test, predictions)


	def next_param(self):
		"""Generate the next parameter set to test"""
		# Adjust chosen parameter
		newparams = copy.copy(self.params)
		change = (self.parameters[self.currentparam].max_value - self.parameters[self.currentparam].min_value) / (2*2**(self.parameters[self.currentparam].stages-self.stages)) # Starting at 1/2 our range, scale change down as a fraction of self.stages
		newparams[self.currentparam] = self.parameters[self.currentparam].adjust_value(newparams[self.currentparam],change)
		gc.collect()
		return newparams

	def parameter_space_search(self, recovered = False):
		"""Navigate through parameter space in half steps until reaching resolution"""
		logging.info(f"Beginning Parameter Space Search...")

		if(not recovered):
			logging.info(f"Testing initial model...")
			self.score = self.cross_validate_model(self.params)
			logging.info(f"Starting Score: {self.score}")

		while self.stages >= 0:
			retune = False

			# Determine which param will be tested
			while(self.currentparam < self.param_count):
				
				logging.info(f"Status Update - Score: {self.score}; Parameters: {self.params}; Currentparam: {self.currentparam}; Direction: {self.parameters[self.currentparam].direction}; Stage: {self.stages};")
				# Have we reached a valid self.stages for this parameter?
				if(self.parameters[self.currentparam].stages >= self.stages):
					
					# If we are at one of our limits, make sure we don't perform a redundant cycle
					if((self.params[self.currentparam] == self.parameters[self.currentparam].min_value) and (self.parameters[self.currentparam].direction == False)):
						# We are at min and trying to go down, move on to the next parameter
						self.currentparam+=1

					else:
						if((self.params[self.currentparam] == self.parameters[self.currentparam].max_value) and (self.parameters[self.currentparam].direction == True)):
							# We are at max and trying to go up, move on to going down
							self.parameters[self.currentparam].direction = False

						# Update parameter for next trial
						newparams = self.next_param()
						logging.debug(f"Trying {self.parameters[self.currentparam].name} from {self.params[self.currentparam]} to {newparams[self.currentparam]}")

						# Run model using new parameter set
						newscore = self.cross_validate_model(newparams)
						gc.collect()

						# Did we get better or worse?
						if(newscore > self.score): # Yes
							# Save the new value
							self.params[self.currentparam] = newparams[self.currentparam]

							# Update Score
							self.score = newscore
							logging.info(f"Improvement found, new high score: {self.score}")

							# Jump to the next parameter (prevents us from repeating our initial state in next iteration if Direction equal true)
							self.currentparam+=1

							# Are we in final tuning?
							if(self.stages == 0):
								retune = True
						else:
							# Adjust for next iteration
							if(self.parameters[self.currentparam].direction == True):
								self.parameters[self.currentparam].direction = False
							else:
								self.currentparam+=1
						
				else:
					self.currentparam+=1

			# Reset for next loop and update stage
			self.currentparam = 0
			for x in self.parameters:
				x.direction = True

			# If we are at the final tuning stage, do not finish until we make it through a full loop with no parameter changes
			if(not retune):
				self.stages-=1
			else:
				if(self.maxrefinements != None):
					self.maxrefinements -= 1

			# Check special exit conditions at the end of each stage
			if self.check_special_exit_conditions():
				self.currentparam = self.param_count
				self.stages = -1

	def run(self, randomstart=False, maxtime=None, maxrefinements=None, recover = False, testonly = False):
		"""Run the optimizer and provide the user with the results"""
		logging.info(f"Beginning run:\n{self.__str__()}")

		# Check that starting conditions are valid, if not exit function
		if self.invariant():
			return self.params, self.score

		# Set up initial values for progress tracking
		self.starttime = time.time()
		self.maxtime = maxtime
		self.maxrefinements = maxrefinements

		for param in self.parameters:
			if randomstart:
				self.params.append(param.round_to_resolution(random.uniform(param.min_value, param.max_value)))
			else:
				self.params.append(param.round_to_resolution((param.max_value - param.min_value) / 2))

		# Are we doing the full run or do we just want a learning curve?
		if(testonly == False):
			# Recovery
			if(recover):
				if(not self.recover_execution()):
					logging.error("Recovery Failed.")
					recover = False

			# Run parameter space search
			try:
				self.parameter_space_search(recovered = recover)
			except MemoryError:
				logging.error("MemoryError occurred: Not enough memory available.")
				sys.exit(1)
			except Exception as e:
				logging.error(f"Unexpected error occurred: {e}")
				sys.exit(1)

		logging.info(f"Parameters converged, performing final scoring...\n")

		self.learning_curve()

		logging.info(f"Final Parameters:\n{self.params}")

		# Record total run time
		runtime = time.time() - self.starttime
		logging.info(f"Total runtime: {runtime}")

		return self.params, self.score

	def learning_curve(self):
		"""Generate learning curve to assess the final model"""

		sample_sizes = np.linspace(self.sample_data.shape[0] * (1 - self.test_split) / 10, self.sample_data.shape[0] * (1 - self.test_split), 10, dtype=int)
		train_scores = []
		test_scores = []
		validation_scores = []

		for sample_size in sample_sizes:
			splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
			train_indices, test_indices = next(splitter.split(self.sample_data, self.sample_class))

			train_data = self.sample_data[train_indices]
			train_labels = self.sample_class[train_indices]

			test_data = self.sample_data[test_indices]
			test_labels = self.sample_class[test_indices]

			m = self.model(self.params, self.sample_data.shape[1:])
			m.compile(loss=self.lossf, metrics=[self.metric])

			early_stopping = EarlyStopping(monitor='val_loss', patience=5)
			m.fit(train_data, to_categorical(train_labels), verbose=0, epochs=self.epochs, callbacks=[early_stopping], validation_split=self.test_split, batch_size=self.batch_size)
			gc.collect()

			train_scores.append(self.test_model(m, train_data, train_labels))
			test_scores.append(self.test_model(m, test_data, test_labels))
			validation_scores.append(self.test_model(m, self.validation_data, self.validation_class))

			gc.collect()

		logging.info("Learning Curve Data:")
		logging.info("Sample Sizes:       " + " ".join([f"{size:8d}" for size in sample_sizes]))
		logging.info("Train Scores:       " + " ".join([f"{score:8.5f}" for score in train_scores]))
		logging.info("Test Scores:        " + " ".join([f"{score:8.5f}" for score in test_scores]))
		logging.info("Validation Scores:  " + " ".join([f"{score:8.5f}" for score in validation_scores]))

		plt.plot(sample_sizes, train_scores, label='Train')
		plt.plot(sample_sizes, test_scores, label='Test')
		plt.plot(sample_sizes, validation_scores, label='Validation')

		plt.title('Learning Curve')
		plt.xlabel('Sample Size')
		plt.ylabel('Score')
		plt.legend(loc='best')

		plot_filename = os.path.join(self.log_directory, self.log_filename + '.png')
		plt.savefig(plot_filename)
		plt.close()

	def recover_execution(self, log_file_path=None):
		if log_file_path is None:
			log_files = glob.glob(os.path.join(self.log_directory, "*.log"))
			log_files_sorted = sorted(log_files, key=os.path.getctime, reverse=True)
			if len(log_files_sorted) >= 2:
				log_file_path = log_files_sorted[1]
			else:
				logging.error("Unable to identify log file for recovery.")
				return False

		logging.info(f"Recovering from: {log_file_path}")

		with open(log_file_path, 'r') as log_file:
			log_lines = log_file.readlines()

		if any("Final Parameters:" in line for line in log_lines):
			logging.error("Logfile is complete, aborting recovery.")
			return False

		status_update_line = None
		for line in reversed(log_lines):
			if "Status Update - Score:" in line:
				status_update_line = line
				break

		if status_update_line is None:
			logging.error("No status update line found, aborting recovery.")
			return False

		self.score = float(re.search(r"Score: ([-+]?\d*\.\d+|\d+)", status_update_line).group(1))
		self.params = [float(x) if '.' in x else int(x) for x in re.findall(r'Parameters: \[([^\]]+)\]', status_update_line)[0].split(',')]
		self.currentparam = int(re.search(r"Currentparam: (\d+)", status_update_line).group(1))
		self.parameters[self.currentparam].direction = True if re.search(r"Direction: (True|False)", status_update_line).group(1) == 'True' else False
		self.stages = int(re.search(r"Stage: (\d+)", status_update_line).group(1))

		logging.info(f"Recovered score: {self.score}")
		logging.info(f"Recovered params: {self.params}")
		logging.info(f"Recovered currentparam: {self.currentparam}")
		logging.info(f"Recovered direction: {self.parameters[self.currentparam].direction}")
		logging.info(f"Recovered stages: {self.stages}")

		return True