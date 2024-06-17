import sys
sys.path.insert(1, '..')

from DataManager import *
from Regression import BinaryRegression
import pandas as pd
from sklearn.metrics import accuracy_score
from Utils import KFold

dataset = Data(pd.read_csv("w3classif.csv", names=['x','y','class']), err_strategy='remove')
dataset.make_categorical('class')

dataset.normalise_numeric()

# dataset.plot_against('x', 'y', categories='class')

# split into inputs and outputs
datasetX, datasetY = dataset.input_output_split(outputs=['class'])

linear_regression_model = BinaryRegression()
linear_regression_model.set_err_function(accuracy_score)
# linear_regression_model.set_regularisation('l1')

linear_regression_model.add_training_data(datasetX, datasetY)

folder = KFold(linear_regression_model, 30, datasetX, datasetY)
folder.fit_all()

training_evaluations = folder.evaluate_training_performances()
testing_evaluations = folder.evaluate_testing_performances()

print("Training Performance", training_evaluations)
print("Testing Performance", testing_evaluations)

print("Training Performance", folder.aggregate_testing_performance())
print("Testing Performance", folder.aggregate_training_performance())