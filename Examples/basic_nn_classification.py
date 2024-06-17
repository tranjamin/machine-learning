import sys
sys.path.insert(1, '..')

from DataManager import *
from NeuralNetwork import NeuralNetwork
import pandas as pd
import tensorflow as tf
import keras
import metrics as tf_metrics
import keras.backend as K
import sklearn.metrics as skm

OUTPUTS = ['class']

dataset = Data(pd.read_csv("w3classif.csv", names=['x','y','class']), err_strategy='remove')

dataset.normalise_numeric()

# dataset.make_numerical("Species")

# onehot encode categoricals
dataset = dataset.encode_categoricals()

# split into training and testing dataset
train_dataset, test_dataset = dataset.train_test_split(split_ratio=0.8, shuffle=True)

# split into inputs and outputs
trainX, trainY = train_dataset.input_output_split(outputs=OUTPUTS)
testX, testY = test_dataset.input_output_split(outputs=OUTPUTS)

# trainX = train_dataset.select(['Species', 'Year_start'])
# testX = test_dataset.select(['Species', 'Year_start'])

# define neural network
nn_model = NeuralNetwork()

# add training and testing data to the model
nn_model.add_training_data(trainX, trainY)
nn_model.add_testing_data(testX, testY)

# define the hyperparameters of the model
nn_model.set_epochs(100) # the number of times to go through the entire dataset
nn_model.set_validation_split_size(0.5) # the amount of data to hold out for validation
nn_model.set_batch_size(256) # the amount of data to make computations on
nn_model.set_early_stopping(metric='val_loss') # early stopping requirement
nn_model.set_optimisation('adam') # the optimiser to run
nn_model.set_loss_function('mean_squared_error') # the loss function
nn_model.add_metric('mean_squared_error') # the metrics to score during and after training

# design model architecture
nn_model.add_dense_layer(num_neurons=128, input_layer=True)
nn_model.add_dense_layer(num_neurons=128)
nn_model.add_dropout(0.1)
nn_model.add_dense_layer(num_neurons=128)
nn_model.add_dense_layer(num_neurons=1)
nn_model.compile_model()

nn_model.summary()
nn_model.fit_model()

testing_evaluations = nn_model.evaluate_testing_performance()
training_evaluations = nn_model.evaluate_training_performance()

print("Training Performance", training_evaluations)
print("Testing Performance", testing_evaluations)

nn_model.visualise_training()

input('')