import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt

import tensorflow as tf
import wandb
import keras
from typing import Optional
import os

import DataManager
import IPython
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
        super().__init__()
        self.model: tf.keras.Sequential = None
        self.layers: list[tf.keras.layers.Layer] = []

        self.optimiser: str | tf.keras.optimizers.Optimizer = None
        self.loss_function: str | tf.keras.losses.Loss = None
        self.metrics: list[str | tf.keras.metrics.Metric] = []
        self.callbacks: list[tf.keras.callbacks] = []

        # defaults
        self.epochs: int = 100
        self.batch_size: int = 32
        self.val_split: float = 0.2

        self.x_train: pd.DataFrame = None
        self.y_train: pd.DataFrame = None
        self.x_test: pd.DataFrame = None
        self.y_test: pd.DataFrame = None

        self.is_functional: bool = False
    
    ### GENERAL ###

    def enable_mixed_precision():
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

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

    ### LAYERS ###

    def add_dense_layer(self, num_neurons: int, activation: str = 'relu', input_layer: bool=False, **kwargs) -> None:
        '''
        Add a dense layer to the neural network architecture.
        The first layer of the network should have input_layer=True
        '''
        if input_layer:
            self.layers.append(tf.keras.layers.Dense(num_neurons, activation=activation, 
                input_shape=self.x_train.shape[1:], **kwargs))
        else:
            self.layers.append(tf.keras.layers.Dense(num_neurons, activation=activation, **kwargs))

    def add_dropout(self, dropout_rate: float = 0.1) -> None:
        '''
        Add a dropout layer to the neural network architecture
        '''
        self.layers.append(tf.keras.layers.Dropout(rate=dropout_rate))
    
    def add_conv2D_layer(self, 
            kernel_size: tuple[int, int], 
            filters: int, 
            activation: str ='relu', 
            padding: Optional[str] = "same",
            strides: Optional[str] = (1,1),
            input_layer: bool=False,
            **kwargs):
        if input_layer:
            self.layers.append(tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, padding=padding, strides=strides, input_shape=self.x_train.shape[1:], activation=activation, **kwargs))
        else:
            self.layers.append(tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, padding=padding, strides=strides, activation=activation, **kwargs))

    def add_pooling2D_layer(self, 
            type: str, 
            size: tuple[int, int],
            padding: str = "valid",
            strides: Optional[tuple[int, int]] = None):
        if type == "max":
            self.layers.append(tf.keras.layers.MaxPool2D(size, strides=strides, padding=padding))
        elif type == "average":
            self.layers.append(tf.keras.layers.AveragePooling2D(size, strides=strides, padding=padding))
    
    def add_global_pooling2D_layer(self, type: str):
        if type == "max":
            self.layers.append(tf.keras.layers.GlobalMaxPooling2D())
        elif type == "average":
            self.layers.append(tf.keras.layers.GlobalAveragePooling2D())
    
    def add_flatten_layer(self, input_layer: bool = False):
        if input_layer:
            self.layers.append(tf.keras.layers.Flatten(input_shape=self.x_train.shape[1:]))
        else:
            self.layers.append(tf.keras.layers.Flatten())

    def add_sparse_output_layer(self):
        self.layers.append(tf.keras.layers.Dense(self.y_train.shape[-1], activation='softmax'))

    def add_output_layer(self):
        self.layers.append(tf.keras.layers.Dense(1, activation='relu'))
    
    def add_batch_norm(self):
        self.layers.append(tf.keras.layers.BatchNormalization())
    
    def add_activation(self, activation="relu"):
        self.layers.append(tf.keras.layers.Activation(activation))

    def add_residual_layer(self, filters, activation="relu", size_match=False):
        def functional_res_block(x):
            skip_connection = x

            x = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding="same",
                strides=(2,2) if size_match else (1,1),
                kernel_initializer=tf.keras.initializers.HeNormal()
                )(x)

            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.Activation(activation)(x)
            x = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding="same", 
                kernel_initializer=tf.keras.initializers.HeNormal())(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)

            if size_match:
                skip_connection = tf.keras.layers.Conv2D(filters, kernel_size=(1,1), strides=(2,2),
                kernel_initializer=tf.keras.initializers.HeNormal())(skip_connection)

            x = tf.keras.layers.Add()([x, skip_connection])
            x = tf.keras.layers.Activation(activation)(x)
            return x
    
        self.layers.append(functional_res_block)
    
    def add_input_layer(self):
        self.layers.append(tf.keras.layers.Input(shape=self.x_train.shape[1:]))
    
    def add_generic_layer(self, layer: tf.keras.layers.Layer) -> None:
        '''
        Add a generic layer
        '''
        self.layers.append(layer)

    ### COMPILE PARAMETERS ###

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

    ### CALLBACKS ###

    def set_early_stopping(self, metric: str='val_loss', patience: int =20, min_delta: float =0.01) -> None:
        '''
        Add early stopping to the network.
        '''
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor=metric, 
            patience=patience, 
            min_delta=min_delta
            )
        self.callbacks.append(earlystopping)
    
    def enable_model_checkpoints(self, path: str ="checkpoints", save_best_only: bool =False, **kwargs):
        '''
        Save the entire model to a checkpoint every epoch

        Parameters:
            path: the folder to store the data in
            save_best_only: only saves the best epoch
        '''
        try:
            os.mkdir(path)
        except OSError:
            pass
        
        if save_best_only:
            checkpoints = tf.keras.callbacks.ModelCheckpoint(path + "/best_epoch.keras", **kwargs)
        else:
            checkpoints = tf.keras.callbacks.ModelCheckpoint(path + "/epoch_{epoch}.keras", **kwargs)
        self.callbacks.append(checkpoints)
    
    def load_checkpoint(self, path):
        '''
        Loads the parameters stored by enable_model_checkpoints.

        Parameters:
            path: the filepath of the checkpoint to load
        '''
        self.model.load_weights(path)
    
    def enable_tensorboard(self, path="./tensorboard.keras", **kwargs):
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=path, **kwargs)
        self.callbacks.append(tensorboard)

    def enable_wandb(self, name="wandb-project"):
        wandb.init(
            project = name,
            # config = tf.flags.FLAGS,
            sync_tensorboard=True
        )

    def add_metric(self, metric: str | tf.keras.metrics.Metric | list[str | tf.keras.metrics.Metric]) -> None:
        '''
        Add a performance metric
        '''
        if type(metric) == type([]):
            self.metrics += metric
        else:
            self.metrics.append(metric)

    def compile_model(self, store_initial_values=False) -> None:
        '''
        Compile the neural network layers
        '''
        if not self.is_functional:
            self.model = tf.keras.Sequential(self.layers)
        else:
            inputs = None
            outputs = None
            x = None
            for layer_no, layer in enumerate(self.layers):
                if layer_no == 0:
                    inputs = layer
                elif layer_no == 1:
                    x = layer(inputs)
                elif layer_no == len(self.layers) - 1:
                    outputs = layer(x)
                else:
                    x = layer(x)

            self.model = tf.keras.Model(inputs, outputs)

        self.model.compile(optimizer=self.optimiser, loss=self.loss_function, metrics=self.metrics)
        if store_initial_values:
            self.initial_variable_values = [tf.identity(var) for var in self.model.trainable_variables]

    def summary(self) -> None:
        '''
        Presents a summary of the neural network
        '''
        self.model.summary()

    def fit(self, verbose=2, **kwargs):
        return self.fit_model(verbose=verbose, **kwargs)

    def fit_model(self, verbose=2, **kwargs):
        '''
        analogous to self.fit()
        '''
        timekeeping = monitorTime()
        timekeeping.start_time()
        self.callbacks.append(timekeeping)

        self.history = self.model.fit(self.x_train, self.y_train, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            verbose=verbose,
                            validation_split=self.val_split,
                            callbacks=self.callbacks,
                            **kwargs
                            )
        
        self.history.history['time_elapsed'] = timekeeping.epoch_times
    
    def fit_model_batches(self, batches, verbose=2, **kwargs):
        timekeeping = monitorTime()
        timekeeping.start_time()
        self.callbacks.append(timekeeping)

        self.history = self.model.fit(batches,
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            verbose=verbose,
                            validation_split=self.val_split,
                            callbacks=self.callbacks,
                            **kwargs
                            )
        
        self.history.history['time_elapsed'] = timekeeping.epoch_times
    
    ### FOR INFERENCE ###

    def predict(self, x):
        return self.model.predict(x)
    
    def evaluate_testing_performance(self):
        metric_evaluations = self.model.evaluate(self.x_test, self.y_test, return_dict=True)
        return metric_evaluations

    def evaluate_training_performance(self):
        metric_evaluations = self.model.evaluate(self.x_train, self.y_train, return_dict=True)
        return metric_evaluations

    def visualise_training(self, to_file=False, filename="training_visualisation.png"):
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
        
        if not to_file:
            fig.show()
        else:
            plt.savefig(filename)
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
    def __init__(self):
        super().__init__()
        self.is_functional = True
    
    def generate_functional_model(self):
        inputs = None
        outputs = None
        x = None
        for layer_no, layer in enumerate(self.layers):
            if layer_no == 0:
                inputs = layer
            elif layer_no == 1:
                x = layer(inputs)
            elif layer_no == len(self.layers) - 1:
                outputs = layer(x)
            else:
                x = layer(x)

        self.model = tf.keras.Model(inputs, outputs)
    
    def compile_functional_model(self):
        self.model.compile(optimizer=self.optimiser, loss=self.loss_function, metrics=self.metrics)

class AdversarialNetwork(NeuralNetwork):
    '''
    A subclass of network designed for GANs
    '''
    
    # a selection of 16 data seeds, each with dim 100
    z = tf.random.normal([16, 100])

    def __init__(self):
        super().__init__()
        self.generator: NeuralNetwork = None
        self.critic: NeuralNetwork = None

        # learning records to keep track of
        self.critic_loss: list = []
        self.generator_loss: list = []
    
    def set_generator(self, model: NeuralNetwork):
        '''
        Define the generator of the GAN
        '''
        self.generator = model
    
    def set_critic(self, model: NeuralNetwork):
        '''
        Define the critic of the GAN
        '''
        self.critic = model

    @tf.function
    def training_step(self, images, update_critic=True, update_generator=True):
        '''
        Computes a training step of the GAN

        Parameters:
            images: the batch of real images
            update_critic: whether this iteration should update the critic
            update_generator: whether this iteration should update the generator
        '''

        # generates z values for the generator to map
        noise = tf.random.normal([self.batch_size, 100])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as critic_tape:
            # forward propagate the z values through the generator
            generated_images = self.generator.get_model()(noise, training=True)
            
            # pass in both real and fake data to the critic
            real_output = self.critic.get_model()(images, training=True)
            fake_output = self.critic.get_model()(generated_images, training=True)

            # evaluate the loss functions
            generator_loss = self.generator.loss_function(fake_output)
            critic_loss = self.critic.loss_function(real_output, fake_output)

        # calculate the gradients of both models
        generator_grad = generator_tape.gradient(generator_loss, self.generator.get_model().trainable_variables)
        critic_grad = critic_tape.gradient(critic_loss, self.critic.get_model().trainable_variables)

        # update the models as required
        if update_generator:
            self.generator.optimiser.apply_gradients(zip(generator_grad, self.generator.get_model().trainable_variables))
        if update_critic:
            self.critic.optimiser.apply_gradients(zip(critic_grad, self.critic.get_model().trainable_variables))
    
    def fit(self, dataset, val_dataset, alpha=1):
        '''
        Fit the GAN

        Parameters:
            dataset: the batched dataset of images to process
            val_dataset: the batched dataset used for validation
            alpha: the update ratio of generator/critic. Defines how often each model is updated in proportion to each other
        '''
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")

            # create a progress bar
            progress_bar = tf.keras.utils.Progbar(self.x_train.shape[0], stateful_metrics=None)

            for image_batch in dataset:
                # execute the relevant training step
                if alpha == 1:
                    self.training_step(image_batch)
                elif alpha < 1:
                    self.training_step(image_batch, update_critic=True, update_generator=not epoch % int(1/alpha))
                else:
                    self.training_step(image_batch, update_generator=True, update_critic=not epoch % int(alpha))

                # generate fake data and predict it
                generated = self.generator.get_model()(AdversarialNetwork.z, training=False)
                generated_preds = self.critic.model.predict(generated, verbose='0')

                # predict real data from the validation set
                real_preds = self.critic.model.predict(val_dataset, verbose='0')

                # evaluate the loss functions
                generator_loss = self.generator.loss_function(generated_preds)
                critic_loss = self.critic.loss_function(real_preds, generated_preds)

                # store the values for logging
                self.generator_loss.append(generator_loss)
                self.critic_loss.append(critic_loss)
                
                # display values on progress bar
                values = [("generator_loss", generator_loss), ("critic_loss", critic_loss)]
                progress_bar.add(self.batch_size, values=values)
            
            # generate image to save on each epoch
            IPython.display.clear_output(wait=True)
            self.generate_image(AdversarialNetwork.z)
        
        # generate final image
        IPython.display.clear_output(wait=True)
        self.generate_image(AdversarialNetwork.z)
    
    def generate_image(self, test_input):
        '''
        Generate an array of inputs

        Parameters:
            test_input: the unmapped z-values to be used for generation
        '''

        # predict values
        predictions = self.generator.get_model()(test_input, training=False)

        # plot predictions
        plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig(f'image_{time.time()}.png')
    
    def visualise_training(self, to_file=False):
        '''
        Inherits from NeuralNetwork.visualise_training
        '''

        plot_size = (1, 1)
        fig, plot_axes = plt.subplots(*plot_size)
        fig.suptitle('Learning Curves')

        subplot = plot_axes
        subplot.set(ylabel="loss")
        subplot.plot(self.generator_loss, label="generator")
        subplot.plot(self.critic_loss, label="critic")
        subplot.legend()
        
        if not to_file:
            fig.show()
        else:
            plt.savefig("training_visualisation2.png")
        return