from __future__ import annotations

from sklearn.linear_model import LinearRegression as LinearModel, LogisticRegression as LogisticModel
from sklearn.linear_model import Ridge as RidgeModel, Lasso as LassoModel, MultiTaskLasso as MultiTaskLassoModel, ElasticNet as ElasticNetModel, MultiTaskElasticNet as MultiTaskElasticNetModel
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.base import clone

import matplotlib.pyplot as plt
import pandas as pd
import numpy.typing
from typing import Callable
import numpy as np
from abc import ABC, abstractmethod

import DataManager
import GenericModel

class Regression(GenericModel.GenericModel, ABC):
    '''
    A class which performs various types of regression.
    '''
    def __init__(self):
        super().__init__()
        self.model: LinearModel | LogisticModel = None
        self.err_function: Callable = None
    
    @abstractmethod
    def set_regularisation(self, reg_type: str, alpha: float=1.0, l1_ratio: float=0.5):
        '''
        Adds regularisation to the model.
        '''
    
    def copy(self) -> Regression:
        # create a new version of the same class
        new = type(self)()

        # clone underlying model and copy across fields
        new.model = clone(self.model)
        new.err_function = self.err_function
        new.x_train = self.x_train
        new.x_test = self.x_test
        new.y_train = self.y_train
        new.y_test = self.y_test

        return new

    def fit(self):
        if self.y_train.shape[1] > 1:
            self.model = MultiOutputRegressor(self.model)
        self.model.fit(self.x_train, self.y_train)
    
    def set_err_function(self, err_function: Callable[[numpy.typing.ArrayLike, numpy.typing.ArrayLike], float]) -> None:
        '''
        Sets the error function which is used to analyse the performance of the model.

        err_function must be a function handle which takes in the true and predicted values as an arraylike structure and returns a float 
        '''
        self.err_function = err_function
    
    def predict(self, x, probabilistic=False, target=None) -> list[float]:
        if probabilistic:
            if isinstance(self.model, MultiOutputRegressor):
                return self.get_model().estimators_[target].predict_proba(x)
            else:
                return self.get_model().predict_proba(x)
        else:
            return self.get_model().predict(x)
    
    def get_model(self):
        return self.model
    
    def polynomial_transform(self, degree: int) -> None:
        '''
        Transforms the model's training and testing input datasets to allow for polynomial regression. For an n-degree polynomial transformation, each column is processed into n columns, representing x, x^2, x^3, ..., x^n.

        Should be called after loading in the datasets.
        '''
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.x_train = pd.DataFrame(poly.fit_transform(self.x_train), columns=poly.get_feature_names_out())
        self.x_test = pd.DataFrame(poly.fit_transform(self.x_test), columns=poly.get_feature_names_out())

    def evaluate_training_performance(self, target: str = None, probabilistic=False) -> float:
        if target is None:
            predictions = self.predict(self.x_train, probabilistic=probabilistic)
            training_err = self.err_function(self.y_train, predictions)
        else:
            if probabilistic:
                targeted_preds = self.predict(self.x_train, probabilistic=probabilistic, target=self.y_train.columns.get_loc(target))
                targeted_vals = self.y_train[target]
                training_err = self.err_function(targeted_vals, targeted_preds)
            else:
                targeted_preds = self.predict(self.x_train, probabilistic=probabilistic)[:, self.y_train.columns.get_loc(target)]
                targeted_vals = self.y_train[target]
                training_err = self.err_function(targeted_vals, targeted_preds)
        return training_err

    def evaluate_testing_performance(self, target: str = None, probabilistic=False) -> float:
        if target is None:
            predictions = self.predict(self.x_test, probabilistic=probabilistic)
            testing_err = self.err_function(self.y_test, predictions)
        else:
            if probabilistic:
                targeted_preds = self.predict(self.x_test, probabilistic=probabilistic, target=self.y_train.columns.get_loc(target))
                targeted_vals = self.y_test[target]
                testing_err = self.err_function(targeted_vals, targeted_preds)
            else:
                targeted_preds = self.predict(self.x_test, probabilistic=probabilistic)[:, self.y_test.columns.get_loc(target)]
                targeted_vals = self.y_test[target]
                testing_err = self.err_function(targeted_vals, targeted_preds)
        return testing_err

    def get_coefficients(self, out_feature: int=0, target: int = 0) -> dict[str, float]:
        '''
        Get the trained coefficients of the model, including the bias. Returns the coefficients in a dictionary form, keyed by the feature names.

        for multi-output applications, out_feature controls which output to display the parameters for. Outputs are indexed [0..num_output_features] according to their position in the original dataset.
        '''
        if isinstance(self.model, MultiOutputRegressor):
            model = self.model.estimators_[target]
        else:
            model = self.model
        theta = {}
        if model.intercept_.ndim == 1:
            theta['bias'] = model.intercept_[0]
        else:
            theta['bias'] = model.intercept_[out_feature]
        for ind, feature_name in enumerate(model.feature_names_in_):
            if model.coef_.ndim == 1:
                theta[feature_name] = model.coef_[ind]
            else:
                theta[feature_name] = model.coef_[out_feature][ind]
        return theta
class LinearRegression(Regression):
    '''
    A class specifically tailored to perform linear regression. Designed to be used with the DataManager module.

    Example usage:

    # load and preprocess data using the DataManager module
    
    from sklearn.metrics import mean_squared_error

    linear_regression_model = LinearRegression()
    linear_regression_model.set_err_function(mean_squared_error) # define the error function
    linear_regression_model.set_regularisation('l1') # optionally add regularisation

    linear_regression_model.add_training_data(trainX, trainY) # add training data
    linear_regression_model.add_testing_data(testX, testY) # add testing data
    linear_regression_model.polynomial_transform(3) # optionally preprocess the data to do polynomial regression
    linear_regression_model.fit() # fit model

    # get some statistics and data
    print("Coefficients: ", linear_regression_model.get_coefficients())
    print("Training Error: ", linear_regression_model.evaluate_training_performance())
    print("Testing Error: ", linear_regression_model.evaluate_testing_performance())

    # plot results
    linear_regression_model.plot_training_predictions(feature_name='x')
    linear_regression_model.plot_testing_predictions(feature_name='x')
    '''
    def __init__(self):
        super().__init__()
        self.model = LinearModel()
        self.err_function = mean_squared_error # default
        self.regularisation_type = None
    
    def augment_hyperparam(self, param: str, value):
        if param == 'regularisation':
            self.set_regularisation(value)
        if param == 'alpha':
            self.set_regularisation(self.regularisation_type, alpha=value)

    def set_regularisation(self, reg_type: str, alpha: float=1.0, l1_ratio: float=0.5):
        '''
        Adds regularisation to the model.

        reg_type must be one of 'l1'/'Lasso', 'l2'/'Ridge', 'MultiTaskLasso', 'ElasticNet'/'Elastic', 'MultiTaskElasticNet'
            and defines the type of regularisation being used
        alpha defines the strength of the regularisation
        l1_ratio defines how much to prioritise l1 over l2, in regularisation methods that use both

        '''
        if reg_type == 'l1' or reg_type == 'Lasso':
            self.model = LassoModel(alpha=alpha)
        elif reg_type == 'l2' or reg_type == 'Ridge':
            self.model = RidgeModel(alpha=alpha)
        elif reg_type == 'MultiTaskLasso':
            self.model = MultiTaskLassoModel(alpha=alpha)
        elif reg_type == 'ElasticNet' or reg_type == 'Elastic':
            self.model = ElasticNetModel(alpha=alpha, l1_ratio=l1_ratio)
        elif reg_type == 'MultiTaskElasticNet':
            self.model = MultiTaskElasticNetModel(alpha=alpha, l1_ratio=l1_ratio)
        
        self.regularisation_type = reg_type

    def plot_training_predictions(self, feature_name: str=None) -> None:
        '''
        Plot the true and predicted values of the training dataset.

        feature_name is the name of the feature to plot the outputs against. If not given, defaults to the first column in the dataset.
        '''
        x = self.x_train.iloc[:, 0] if feature_name is None else self.x_train[feature_name]
        plt.scatter(x, self.predict(self.x_train), label='Predictions')
        plt.scatter(x, self.y_train, label='Actual')
        plt.legend()
        plt.xlabel(feature_name if feature_name is not None else self.x_train.columns[0])
        plt.ylabel(self.y_train.columns[0])
        plt.title("Training Performance")
        plt.show()
        
    def plot_testing_predictions(self, feature_name: str=None) -> None:
        '''
        Plot the true and predicted values of the testing dataset.

        feature_name is the name of the feature to plot the outputs against. If not given, defaults to the first column in the dataset.
        '''
        x = self.x_test[:, 0] if feature_name is None else self.x_test[feature_name]
        plt.scatter(x, self.predict(self.x_test), label="Predictions")
        plt.scatter(x, self.y_test, label="Actual")
        plt.legend()
        plt.xlabel(feature_name if feature_name is not None else self.x_test.columns[0])
        plt.ylabel(self.y_test.columns[0])
        plt.title("Testing Performance")
        plt.show()

class BinaryRegression(Regression):
    '''
    A class specifically tailored to perform 2-class classification (logistic regression)
    '''
    def __init__(self):
        super().__init__()
        self.model = LogisticModel()
    
    def set_regularisation(self, penalty, C=1.0, l1_ratio=None):
        '''
        Adds regularisation to the moodel, using the 'saga' solver

        penalty is the type of regularisation to be added
        C is the inverse of regularisation strength
        l1_ratio defines how much to prioritise l1 over l2, in methods that use both

        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        '''
        self.model = LogisticModel(penalty=penalty, C=C, l1_ratio=l1_ratio, solver='saga')

    def plot_training_predictions(self, x_name: str=None, y_name: str=None) -> None:
        '''
        Plot the true and predicted classes of the training dataset.

        x_name,y_name are the names of the features to plot the classes against. If not given, defaults to the first and second column in the dataset.
        '''
        x = self.x_train.iloc[:, 0] if x_name is None else self.x_train[x_name]
        y = self.x_train.iloc[:, 1] if y_name is None else self.x_train[y_name]

        preds = self.predict(self.x_train)
        actual = np.reshape(self.y_train.iloc[:, 0], preds.shape)
        plt.scatter(x, y, c=preds, label='Predictions', marker='o')
        plt.scatter(x, y, c=actual, label='Actual', marker='.')
        plt.legend()
        plt.xlabel(x_name if x_name is not None else self.x_train.columns[0])
        plt.ylabel(y_name if y_name is not None else self.x_train.columns[1])
        plt.title("Training Performance")
        plt.show()
        
    def plot_testing_predictions(self, x_name: str=None, y_name: str=None) -> None:
        '''
        Plot the true and predicted classes of the testing dataset.

        x_name,y_name are the names of the features to plot the classes against. If not given, defaults to the first and second column in the dataset.
        '''
        x = self.x_test.iloc[:, 0] if x_name is None else self.x_test[x_name]
        y = self.x_test.iloc[:, 1] if y_name is None else self.x_test[y_name]

        preds = self.predict(self.x_test)
        actual = np.reshape(self.y_test.iloc[:, 0], preds.shape)
        plt.scatter(x, y, c=preds, label='Predictions', marker='o')
        plt.scatter(x, y, c=actual, label='Actual', marker='.')
        plt.legend()
        plt.xlabel(x_name if x_name is not None else self.x_test.columns[0])
        plt.ylabel(y_name if y_name is not None else self.x_test.columns[1])
        plt.title("Testing Performance")
        plt.show()
    
class MulticlassRegression(Regression):
    def __init__(self):
        super().__init__()
        self.model = LogisticModel()