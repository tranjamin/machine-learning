import sys
sys.path.insert(1, '..')

from DataManager import *
from Regression import BinaryRegression
import pandas as pd
from sklearn.metrics import accuracy_score

dataset = Data(pd.read_csv("w3classif.csv", names=['x','y','class']), err_strategy='remove')
dataset.make_categorical('class')
dataset.normalise_numeric()

# dataset.plot_against('x', 'y', categories='class')

# onehot encode categoricals

# split into training and testing dataset
train_dataset, test_dataset = dataset.train_test_split(split_ratio=0.7, shuffle=True)

# split into inputs and outputs
trainX, trainY = train_dataset.input_output_split(outputs=['class'])
testX, testY = test_dataset.input_output_split(outputs=['class'])

linear_regression_model = BinaryRegression()
linear_regression_model.set_err_function(accuracy_score)
# linear_regression_model.set_regularisation('l1')

linear_regression_model.add_training_data(trainX, trainY)
linear_regression_model.add_testing_data(testX, testY)
linear_regression_model.fit()

print("Coefficients: ", linear_regression_model.get_coefficients())
print("Training Error: ", linear_regression_model.evaluate_training_performance())
print("Testing Error: ", linear_regression_model.evaluate_testing_performance())

linear_regression_model.plot_training_predictions('x', 'y')
linear_regression_model.plot_testing_predictions('x', 'y')
