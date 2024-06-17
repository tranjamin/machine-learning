import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
import keras.backend as K

import DataManager
from Utils import GenericModel

class monitorTime(tf.keras.callbacks.Callback):
    '''
    Class which monitors the time usage of the training
    '''
    def __init__(self):
        self.epoch_times = []
    
    def start_time(self):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_times.append(time.time() - self.start_time)

class NeuralNetwork(GenericModel):
    '''
    A class which performs (sequential) neural networks
    '''
    def __init__(self):
        self.model: tf.keras.Sequential = None
        self.layers: list[tf.keras.layers.Layer] = []

        self.optimiser: str | tf.keras.optimizers.Optimizer = None
        self.loss_function: str | tf.keras.losses.Loss = None
        self.metrics: list[str | tf.keras.metrics.Metric] = []
        self.earlystopping: tf.keras.callbacks.EarlyStopping | None = None

        # defaults
        self.epochs: int = 100
        self.batch_size: int = 32
        self.val_split: float = 0.2

        self.x_train: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.x_test: pd.DataFrame = None
        self.y_test: pd.DataFrame = None

        self.is_functional: bool = False
    
    def make_functional(self):
        self.is_functional = True

    def set_epochs(self, epochs: int) -> None:
        '''
        Set the number of epochs to run through
        '''
        self.epochs = epochs
    
    def augment_hyperparam(self, param, value):
        pass

    def get_model(self):
        return self.model
    
    def copy(self):
        new = NeuralNetwork()

        # duplicate layers        
        new.layers = []
        for layer in self.layers:
            new.layers.append(type(layer).from_config(layer.get_config()))

        # duplicate optimiser
        if isinstance(self.optimiser, str):
            new.optimiser = self.optimiser
        else:
            new.optimiser = type(self.optimiser).from_config(self.optimiser.get_config())
        
        # duplicate loss function
        if isinstance(self.loss_function, str):
            new.loss_function = self.loss_function
        elif isinstance(self.loss_function, list):
            new.loss_function = [type(x).from_config(x.get_config()) for x in self.loss_function]
        else:
            new.loss_function = type(self.loss_function).from_config(self.loss_function.get_config())
        
        # duplicate all metrics
        new.metrics = []
        for metric in self.metrics:
            if isinstance(metric, str):
                new.metrics.append(metric)
            else:
                new.metrics.append(type(metric).from_config(metric.get_config()))

        # duplicate [NEED-TO-DO] earlystopping
        new.earlystopping = self.earlystopping

        # copy over member properties
        new.epochs = self.epochs
        new.batch_size = self.batch_size
        new.val_split = self.val_split

        new.x_train = self.x_train
        new.y_train = self.y_train
        new.x_test = self.x_test
        new.y_test = self.y_test

        new.is_functional = self.is_functional
        if self.model is not None:
            new.model = tf.keras.models.clone_model(self.model)

        # compile the model
        new.compile_model()
        
        return new

    def set_batch_size(self, batch_size: int) -> None:
        '''
        Set the batch size of the network
        '''
        self.batch_size = batch_size
    
    def set_validation_split_size(self, size: int) -> None:
        '''
        Set the validation split size of the network
        '''
        self.val_split = size

    def add_dense_layer(self, num_neurons: int, activation: str = 'relu', input_layer: bool=False) -> None:
        '''
        Add a dense layer to the neural network architecture.
        The first layer of the network should have input_layer=True
        '''
        if input_layer:
            num_features = self.x_train.shape[1]
            self.layers.append(tf.keras.layers.Dense(num_neurons, activation=activation, input_shape=(num_features,)))
        else:
            self.layers.append(tf.keras.layers.Dense(num_neurons, activation=activation))

    def add_dropout(self, dropout_rate: float = 0.1) -> None:
        '''
        Add a dropout layer to the neural network architecture
        '''
        self.layers.append(tf.keras.layers.Dropout(rate=dropout_rate))
    
    def add_generic_layer(self, layer: tf.keras.layers.Layer) -> None:
        '''
        Add a generic layer
        '''
        self.layers.append(layer)

    def set_loss_function(self, loss_function: str | tf.keras.losses.Loss) -> None:
        '''
        Set the loss function
        '''
        self.loss_function = loss_function

    def set_optimisation(self, optimiser: str | tf.keras.optimizers.Optimizer) -> None:
        '''
        Set the optimisation
        '''
        self.optimiser = optimiser

    def set_early_stopping(self, metric: str='val_loss', patience: int =20, min_delta: float =0.01) -> None:
        '''
        Add early stopping to the network.
        '''
        self.earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=metric, 
            patience=patience, 
            min_delta=min_delta
            )

    def add_metric(self, metric: str | tf.keras.metrics.Metric) -> None:
        '''
        Add a performance metric
        '''
        self.metrics.append(metric)

    def add_training_data(self, x_train, y_train):
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
    
    def add_testing_data(self, x_test, y_test):
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

    def compile_model(self) -> None:
        '''
        Compile the neural network layers
        '''
        if not self.is_functional:
            self.model = tf.keras.Sequential(self.layers)
        self.model.compile(optimizer=self.optimiser, loss=self.loss_function, metrics=self.metrics)

        self.initial_variable_values = [tf.identity(var) for var in self.model.trainable_variables]

    def summary(self) -> None:
        '''
        Presents a summary of the neural network
        '''
        self.model.summary()

    def fit(self):
        return self.fit_model()

    def fit_model(self):
        '''
        analogous to self.fit()
        '''
        timekeeping = monitorTime()
        timekeeping.start_time()
        self.history = self.model.fit(self.x_train, self.y_train, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            verbose=2,
                            validation_split=self.val_split,
                            callbacks=([self.earlystopping, timekeeping] if self.earlystopping is not None else [timekeeping])
                            )
        
        self.history.history['time_elapsed'] = timekeeping.epoch_times
    
    def predict(self, x):
        return self.model.predict(x)
    
    def evaluate_testing_performance(self):
        metric_evaluations = self.model.evaluate(self.x_test, self.y_test, return_dict=True)
        return metric_evaluations

    def evaluate_training_performance(self):
        metric_evaluations = self.model.evaluate(self.x_train, self.y_train, return_dict=True)
        return metric_evaluations

    def visualise_training(self):
        '''
        visualise the learning curve of a network
        '''
        metrics = list(self.history.history.keys())
        for metric_name in metrics.copy():
            if self.history.history.get('val_' + metric_name) is not None:
                metrics.pop(metrics.index('val_' + metric_name))
        
        plot_size = (math.ceil(math.sqrt(len(metrics))),math.ceil(math.sqrt(len(metrics))))
        fig, plot_axes = plt.subplots(*plot_size)
        fig.suptitle('Learning Curves')

        for i in range(len(metrics)):
            metric_name = metrics[i]
            subplot = plot_axes[i // plot_size[0],i % plot_size[0]]
            subplot.set(ylabel=metric_name)
            subplot.plot(self.history.history[metric_name], label='train')
            if self.history.history.get('val_' + metric_name) is not None:
                subplot.plot(self.history.history['val_' + metric_name], label='val')
            subplot.legend()
        
        fig.show()
        return

    def get_dominance(self, type='connection_weights'):
        if type == 'connection_weights':
            layer_params: list[tf.Variable] = self.get_model().trainable_variables
            layer_importances: list[list[float]] = []
            for i in range(len(layer_params) - 2, -1, -2):
                kernel_layer = layer_params[i] # kernel layers starting back to front
                kernel_layer = np.array(kernel_layer)

                if i == len(layer_params) - 2: # output weights just sum
                    importances = kernel_layer.sum(axis=-1)
                    layer_importances.append(importances)
                else:
                    importances = np.matmul(kernel_layer, layer_importances[-1])
                    layer_importances.append(importances)

            input_importances = abs(layer_importances[-1])
            feature_names = self.x_train.columns
            return dict(zip(feature_names, input_importances / input_importances.sum()))
        elif type == 'most_squares':
            layer_params: list[tf.Variable] = self.get_model().trainable_variables
            layer_importances: list[list[float]] = []
            for i in range(len(layer_params) - 2, -1, -2):
                kernel_layer = layer_params[i] # kernel layers starting back to front
                initial_kernel_layer = self.initial_variable_values[i]
                kernel_layer = np.subtract(np.array(kernel_layer), np.array(initial_kernel_layer))
                kernel_layer = np.multiply(kernel_layer, kernel_layer)

                if i == len(layer_params) - 2: # output weights just sum
                    importances = kernel_layer.sum(axis=-1)
                    layer_importances.append(importances)
                else:
                    importances = np.matmul(kernel_layer, layer_importances[-1])
                    layer_importances.append(importances)

            input_importances = abs(layer_importances[-1])
            feature_names = self.x_train.columns
            return dict(zip(feature_names, input_importances / input_importances.sum()))
        elif type == 'garson':
            layer_params: list[tf.Variable] = self.get_model().trainable_variables
            layer_importances: list[list[float]] = []
            for i in range(len(layer_params) - 2, -1, -2):
                kernel_layer = layer_params[i] # kernel layers starting back to front
                kernel_layer = np.array(kernel_layer)
                kernel_layer = np.divide(kernel_layer, np.sum(kernel_layer, axis=0))

                if i == len(layer_params) - 2: # output weights just sum
                    importances = kernel_layer.sum(axis=-1)
                    layer_importances.append(importances)
                else:
                    importances = np.matmul(kernel_layer, layer_importances[-1])
                    layer_importances.append(importances)

            input_importances = abs(layer_importances[-1])
            feature_names = self.x_train.columns
            return dict(zip(feature_names, input_importances / input_importances.sum()))

class FunctionalNetwork(NeuralNetwork):
    pass