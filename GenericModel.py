from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

import DataManager

import pandas as pd
import numpy.typing

class GenericModel(ABC):
    '''
    Generic model that all other models should be extended from.
    '''
    def __init__(self):
        self.x_train: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.x_test: pd.DataFrame = None
        self.y_test: pd.DataFrame = None

    @abstractmethod
    def get_model(self) -> Any:
        '''
        Get the underlying model. Return type is not guaranteed, but typically is either an sklearn or keras model.
        '''

    @abstractmethod
    def copy(self) -> GenericModel:
        '''
        Copy and return a new model. Used for duplicating models.
        '''

    def add_training_data(self, x_train: DataManager.Data | pd.DataFrame, y_train: DataManager.Data | pd.DataFrame) -> None:
        '''
        Loads in the training data into the model. The data can be a DataManager.Data object or a pandas DataFrame. 
        It is not recommended to load data in other formats because it is not supported and errors may occur.
        The DataFrame must have all columns named and be suitably preprocessed into numeric data.
        '''
        try:
            x_train = x_train.get_data()
        except:
            pass

        try:
            y_train = y_train.get_data()
        except:
            pass

        self.x_train = x_train
        self.y_train = y_train

    def add_testing_data(self, x_test: DataManager.Data | pd.DataFrame, y_test: DataManager.Data | pd.DataFrame) -> None:
        '''
        Loads in the testing data into the model. The data can be a DataManager.Data object or a pandas DataFrame. 
        It is not recommended to load data in other formats because it is not supported and errors may occur.
        The DataFrame must have all columns named and be suitably preprocessed into numeric data.
        '''
        try:
            x_test = x_test.get_data()
        except:
            pass
        
        try:
            y_test = y_test.get_data()
        except:
            pass

        self.x_test = x_test
        self.y_test = y_test

    @abstractmethod
    def fit(self) -> None:
        '''
        Fit the model to the training data. Essentially a wrapper for the underlying model.
        Must have added training data.
        '''
    
    @abstractmethod
    def predict(self, x: numpy.typing.ArrayLike, probabilistic=False, target=None) -> list[float]:
        '''
        Predicts the output of an input or series of inputs. x can be anything which can be converted to an array. Returns the predicted values in an array.
        
        For classification models, probabilistic should be set to False to predict hard labels, or True for probabilities.
        For multi-target models, specify the target of the prediction.
        '''


    @abstractmethod
    def augment_hyperparam(self, param: str, value: Any):
        '''
        Augment a hyperparameter of the model. Takes in the parameter as a string, and it is the class's responsibility to parse it.
        '''

    @abstractmethod
    def evaluate_testing_performance(self, target: str = None, probabilistic=False) -> float | dict[str, float]:
        '''
        Evaluate the performance of the model under the testing dataset. Will either return a single value corresponding to the
        loss function, or if multiple metrics are specified will return a dictionary of values.

        For classification models, probabilistic should be set to False to predict hard labels, or True for probabilities.
        For multi-target models, specify the target of the prediction.
        '''
    
    @abstractmethod
    def evaluate_training_performance(self, target: str = None, probabilistic=False) -> float | dict[str, float]:
        '''
        Evaluate the performance of the model under the training dataset. Will either return a single value corresponding to the
        loss function, or if multiple metrics are specified will return a dictionary of values.
        
        For classification models, probabilistic should be set to False to predict hard labels, or True for probabilities
        For multi-target models, specify the target of the prediction.
        '''