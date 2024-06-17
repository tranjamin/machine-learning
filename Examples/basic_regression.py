import sys
sys.path.insert(1, '..')

from DataManager import *
from Regression import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error

dataset = Data(pd.read_csv("w3regr.csv", header=None, names=['x','y']), err_strategy='remove')
dataset.normalise_numeric()

# onehot encode categoricals
dataset = dataset.encode_categoricals()

# split into training and testing dataset
train_dataset, test_dataset = dataset.train_test_split(split_ratio=0.7, shuffle=True)

# split into inputs and outputs
trainX, trainY = train_dataset.input_output_split(outputs=['y'])
testX, testY = test_dataset.input_output_split(outputs=['y'])

linear_regression_model = LinearRegression()
linear_regression_model.set_err_function(mean_squared_error)
linear_regression_model.set_regularisation('l1')

linear_regression_model.add_training_data(trainX, trainY)
linear_regression_model.add_testing_data(testX, testY)
# linear_regression_model.polynomial_transform(30)
linear_regression_model.fit()

print("Coefficients: ", linear_regression_model.get_coefficients())
print("Training Error: ", linear_regression_model.evaluate_training_performance())
print("Testing Error: ", linear_regression_model.evaluate_testing_performance())
linear_regression_model.plot_training_predictions(feature_name='x')
linear_regression_model.plot_testing_predictions(feature_name='x')