import numpy as np
from typing import List, Tuple, cast

import flwr as fl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import random
random.seed(10)

from variables import dataset_name, epochs, batch_size, steps_per_epoch


# Define a Flower client
class CifarClient(fl.client.NumPyClient):
    
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model   = model
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.x_test  = np.array(x_test)
        self.y_test  = np.array(y_test)

    
    def get_parameters(self):
        """Return current weights."""
        
        return self.model.get_weights()

    
    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training examples."""
        
        self.model.set_weights(parameters)
        
        if steps_per_epoch == 0:
            history = self.model.fit(self.x_train, self.y_train, epochs = epochs, batch_size = batch_size)
            
        elif steps_per_epoch == "divide":
            history = self.model.fit(self.x_train, self.y_train, epochs = epochs, batch_size = batch_size,
                                     steps_per_epoch  = len(self.x_train) // batch_size)
            
        else:
            history = self.model.fit(self.x_train, self.y_train, epochs = epochs, batch_size = batch_size,
                                     steps_per_epoch  = steps_per_epoch)
        
        # Return updated model parameters and results
        parameters_prime   = self.model.get_weights()
        
        num_examples_train = len(self.x_train)
        
        results = {
                   "loss"        : history.history["loss"][0],
                   "accuracy"    : history.history["accuracy"][0],
#                    "val_loss"    : history.history["val_loss"][0],
#                    "val_accuracy": history.history["val_accuracy"][0]
                  }

        return parameters_prime, num_examples_train, results

    
    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""

        self.model.set_weights(parameters)
        
        loss, accuracy    = self.model.evaluate(self.x_test, self.y_test)
        
        num_examples_test = len(self.x_test)

        return loss, num_examples_test, {"accuracy": accuracy, "loss": loss}


    
DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def get_model_mnist():
    ## https://keras.io/examples/vision/mnist_convnet/
    
    num_classes = 10
    input_shape = (28, 28, 1)
    
    model = keras.Sequential([
        keras.Input(shape = input_shape),
        layers.Conv2D(32, kernel_size = (3, 3), activation = "relu"),
        layers.MaxPooling2D(pool_size = (2, 2)),
        layers.Conv2D(64, kernel_size = (3, 3), activation = "relu"),
        layers.MaxPooling2D(pool_size = (2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation = "softmax"),
    ])
    
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    return model   


def get_model_cifar10():
    ## https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    
    num_classes = 10
    input_shape = (32, 32, 3)
    
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same', 
                            input_shape = input_shape))
    
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes, activation = 'softmax'))
    
    # compile model
    opt = keras.optimizers.SGD(lr = 0.001, momentum = 0.9)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model



def start_client(dataset: DATASET) -> None:
    """Start a single client with the provided dataset."""
    
    # Load and compile a Keras model
    if dataset_name == "mnist":
        model = get_model_mnist()
    
    elif dataset_name == "cifar10":
        model = get_model_cifar10()

    # Unpack the dataset partition
    (x_train, y_train), (x_test, y_test) = dataset

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("[::]:8080", client = client)
