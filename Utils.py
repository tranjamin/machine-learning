from __future__ import annotations
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from DataManager import Data
from Regression import Regression
from GenericModel import GenericModel

from sklearn.model_selection import KFold as sklearnKFold

class Bagger():
    '''
    A class which can bag models together
    '''
    def __init__(self, model: GenericModel, x_train: Data, y_train: Data, B: int):
        self.x_data: Data = x_train
        self.y_data: Data = y_train
        self.models: list[GenericModel] = []


        for i in range(B):
            # duplicate model
            model_copy = model.copy()

            # bootstrap new data
            seed = random.randint(0, 1000)
            sampled_x = x_train.bootstrap(seed=seed)
            sampled_y = y_train.bootstrap(seed=seed)
            model_copy.add_training_data(sampled_x, sampled_y)

            self.models.append(model_copy)
        
    def fit_all(self) -> None:
        '''
        Fit all instances of the model
        '''
        for model in self.models:
            model.fit()
    
    def out_of_bag_datapoint(self, index: int) -> float:
        '''
        Calculate the out of bag error for a single datapoint
        '''
        num_models = 0
        running_sum = 0

        for model in self.models:
            if index in model.x_train.index:
                num_models += 1
                model.add_testing_data(pd.DataFrame(self.x_data.get_data().loc[index]), pd.DataFrame(self.y_data.get_data().loc[index]))
                running_sum += model.evaluate_testing_performance()
        
        return 0 if not num_models else running_sum / num_models
    
    def out_of_bag_error(self) -> float:
        '''
        Calculate the total out of bag error
        '''
        num_points = 0
        running_sum = 0

        for index in self.x_data.get_data().index:
            running_sum += self.out_of_bag_datapoint(index)
            num_points += 1
        
        return 0 if not num_points else running_sum / num_points

class KFold(GenericModel):
    '''
    Class which performs K-Fold Cross-Validation. Can act as a overall model.
    '''

    def __init__(self, model: GenericModel, n_splits: int, x_data: Data, y_data: Data):
        self.all_models: list[GenericModel] = []
        self.master_model: GenericModel = []

        kf = sklearnKFold(n_splits=n_splits, shuffle=True)
        for (train_index, test_index) in kf.split(x_data.get_data()):
            new_model = model.copy()
            train_foldX = Data(x_data.get_data().loc[train_index])
            train_foldY = Data(y_data.get_data().loc[train_index])
            test_foldX = Data(x_data.get_data().loc[test_index])
            test_foldY = Data(y_data.get_data().loc[test_index])

            new_model.add_training_data(train_foldX, train_foldY)
            new_model.add_testing_data(test_foldX, test_foldY)

            self.all_models.append(new_model)
    
        self.master_model = model
        self.master_model.add_training_data(x_data, y_data)
        
        self.training_performances: float | dict[str, float] = None
        self.testing_performances: float | dict[str, float] = None
    
    def fit(self):
        self.fit_all()

    def fit_all(self):
        '''
        Analogous to self.fit()
        '''
        for model in self.all_models:
            model.fit()
        self.master_model.fit()
    
    def predict(self, data):
        return self.master_model.predict(data)
    
    def evaluate_testing_performances(self, **kwargs) -> list[float | dict[str, float]]:
        '''
        Evaluate the testing performance of each individual fold
        '''
        self.testing_performances = []
        for model in self.all_models:
            self.testing_performances.append(model.evaluate_testing_performance(**kwargs))
        return self.testing_performances

    def evaluate_training_performances(self, **kwargs) -> list[float | dict[str, float]]:
        '''
        Evaluate the training performance of each individual fold
        '''
        self.training_performances = []
        for model in self.all_models:
            self.training_performances.append(model.evaluate_training_performance(**kwargs))
        return self.training_performances

    def aggregate_testing_performance(self, **kwargs) -> float | dict[str, float]:
        '''
        Analogous to self.evaluate_testing_performance()
        '''
        self.evaluate_testing_performances(**kwargs)

        if isinstance(self.testing_performances[0], dict):
            self.aggregated = dict()
            for key in self.testing_performances[0].keys():
                individual_aggregate = 0
                for fold in self.testing_performances:
                    individual_aggregate += fold[key]
                individual_aggregate /= len(self.testing_performances)
                self.aggregated[key] = individual_aggregate
            
            return self.aggregated

        else:
            return sum(self.testing_performances) / len(self.testing_performances)

    def aggregate_training_performance(self, **kwargs) -> float | dict[str, float]:
        '''
        Analogous to self.evaluate_training_performance()
        '''
        self.evaluate_training_performances(**kwargs)

        if isinstance(self.training_performances[0], dict):
            self.aggregated = dict()
            for key in self.training_performances[0].keys():
                individual_aggregate = 0
                for fold in self.training_performances:
                    individual_aggregate += fold[key]
                individual_aggregate /= len(self.training_performances)
                self.aggregated[key] = individual_aggregate
            
            return self.aggregated

        else:
            return sum(self.training_performances) / len(self.training_performances)
    
    def evaluate_testing_performance(self, target: str = None, probabilistic=False):
        return self.aggregate_testing_performance(target=target, probabilistic=probabilistic)
    
    def evaluate_training_performance(self, target: str = None, probabilistic=False):
        return self.aggregate_training_performance(target=target, probabilistic=probabilistic)

    def augment_hyperparam(self, param: str, value: random.Any):
        pass

    def get_model(self):
        return self.master_model
    
    def copy(self):
        pass

    def add_testing_data(self, x_test: Data | pd.DataFrame, y_test: Data | pd.DataFrame) -> None:
        self.master_model.add_testing_data(x_test, y_test)
    
    def add_training_data(self, x_train: Data | pd.DataFrame, y_train: Data | pd.DataFrame) -> None:
        self.__init__(self.master_model, len(self.all_models), x_train, y_train)

class HyperparameterTuner():
    '''
    Class which performs hyperparameter tuning.
    '''
    def __init__(self, model):
        self.parameters: dict[str, list] = {}
        self.model: GenericModel = model
        self.results_params: list[str] = []
        self.results_models: list = []
    
    def get_hyperparameter_values(self, param: str):
        return self.parameters[param]

    def add_hyperparameter(self, param: str, options: list) -> None:
        self.parameters[param] = options
    
    def add_hyperparameter_fromrange(self, param: str, start: int, stop: int, step:int=1) -> None:
        self.add_hyperparameter(param, list(np.arange(start, stop, step)))
    
    def tune_hyperparameters(self, params: str | list[str], specific_model=None) -> tuple[list[str], list[GenericModel]]:
        if isinstance(params, str) or (isinstance(params, list) and len(params) == 1):
            if isinstance(params, list):
                params = params[0]

            fitted_models = []
            for value in self.parameters[params]:
                copied = (self.model if specific_model is None else specific_model).copy()
                copied.augment_hyperparam(params, value)
                copied.fit()
                fitted_models.append(copied)

            self.results_params = [params]
            self.results_models = fitted_models
            return [params], fitted_models
        else:
            fitted_models = []
            sequential_names = []
            param_name = params[0]
            for value in self.parameters[param_name]:
                copied = self.model.copy()
                copied.augment_hyperparam(params, value)

                recurrent_names, recurrent_models = self.tune_hyperparameters(params=params[1:], specific_model=copied)
                fitted_models.append(recurrent_models)
                sequential_names = [param_name] + recurrent_names
            
            self.results_params = sequential_names
            self.results_models = fitted_models
            return sequential_names, fitted_models

    def visualise_1d_testing(self, param: str, series: str =None):
        x_data = [str(x) for x in self.get_hyperparameter_values(param)]
        y_data = [model.evaluate_testing_performance() for model in self.results_models]

        plt.title(f'Hyperparameter Tuning of {param}')
        plt.xlabel(param)
        plt.ylabel('Testing Performance')
        plt.plot(x_data, y_data)
        plt.show()

    def visualise_1d_training(self, param: str, series: str =None):
        x_data = [str(x) for x in self.get_hyperparameter_values(param)]
        y_data = [model.evaluate_training_performance() for model in self.results_models]

        plt.title(f'Hyperparameter Tuning of {param}')
        plt.xlabel(param)
        plt.ylabel('Training Performance')
        plt.plot(x_data, y_data)
        plt.show()

    def visualise_1d(self, param: str):
        x_data = [str(x) for x in self.get_hyperparameter_values(param)]
        y_data_training = [model.evaluate_training_performance() for model in self.results_models]
        y_data_testing = [model.evaluate_testing_performance() for model in self.results_models]

        plt.title(f'Hyperparameter Tuning of {param}')
        plt.xlabel(param)
        plt.ylabel('Performance')
        plt.plot(x_data, y_data_training, label='Training')
        plt.plot(x_data, y_data_testing, label='Testing')
        plt.legend()
        plt.show()

    def visualise_2d_training(self):
        x_data = [str(x) for x in self.get_hyperparameter_values(self.results_params[1])]
        y_data = [str(y) for y in self.get_hyperparameter_values(self.results_params[0])]

        xy_data = [[model.evaluate_training_performance() for model in arr] for arr in self.results_models]

        ax = sns.heatmap(xy_data, xticklabels=x_data, yticklabels=y_data, annot=True)
        ax.set(xlabel=self.results_params[0], ylabel=self.results_params[1])

class Repeater():
    pass