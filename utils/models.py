import matplotlib.pyplot as plt
import numpy as np
import os
import joblib

class Perceptron:
    """
    Initializes a Perceptron object.

    Parameters:
    - eta (float): The learning rate.
    - epochs (int): The number of epochs (complete cycles of forward propagation + backward propagation).

    Author: Dipendra Pratap Singh

    """
    def __init__(self, eta: float = None, epochs: int = None):
        """
        Initializes a Perceptron object.

        Parameters:
        - eta (float): The learning rate.
        - epochs (int): The number of epochs (complete cycles of forward propagation + backward propagation).

        """
        try:
            self.weights = np.random.randn(3) * 1e-4  # giving small random weights
            training = (eta is not None) and (epochs is not None)

            if training:
                print(f"Initial weights before training:\n{self.weights}")
            self.eta = eta
            self.epochs = epochs

        except Exception as e:
            raise e

    def _z_outcome(self, inputs_with_bias, weights):
        """
        Calculates the dot product of inputs and weights.

        Parameters:
        - inputs_with_bias (ndarray): Inputs with bias column.
        - weights (ndarray): Model weights.

        Returns:
        - ndarray: The dot product of inputs and weights.

        """
        try:
            return np.dot(inputs_with_bias, weights)
        except Exception as e:
            raise e

    def activation_function(self, z):
        """
        Applies the activation function on the given input.

        Parameters:
        - z (ndarray): Input values.

        Returns:
        - ndarray: The output after applying the activation function.

        """
        try:
            return np.where(z > 0, 1, 0)
        except Exception as e:
            raise e

    def fit(self, x, y):
        """
        Fits the perceptron model to the training data.

        Parameters:
        - x (ndarray): Training input data.
        - y (ndarray): Target labels.

        """
        try:
            self.x = x
            self.y = y

            x_with_bias = np.c_[self.x, -np.ones((len(self.x), 1))]
            print(f"X with bias:\n{x_with_bias}")

            for epoch in range(self.epochs):
                print("--" * 10)
                print(f"For epoch >> {epoch}")
                print("--" * 10)

                z = self._z_outcome(x_with_bias, self.weights)
                y_hat = self.activation_function(z)
                print(f"Predicted value after forward pass:\n{y_hat}")

                self.error = self.y - y_hat
                print(f"Error:\n{self.error}")

                self.weights = self.weights + self.eta * np.dot(x_with_bias.T, self.error)
                print(f"Updated weights after epoch: {epoch + 1}/{self.epochs}:\n{self.weights}")
                print("====" * 10)

        except Exception as e:
            raise e

    def predict(self, x):
        """
        Predicts the target labels for the given input data.

        Parameters:
        - x (ndarray): Input data.

        Returns:
        - ndarray: Predicted target labels.

        """
        try:
            x_with_bias = np.c_[x, -np.ones((len(x), 1))]
            z = self._z_outcome(x_with_bias, self.weights)
            return self.activation_function(z)

        except Exception as e:
            raise e

    def total_loss(self):
        """
        Computes the total loss.

        Returns:
        - float: The total loss.

        """
        try:
            total_loss = np.sum(self.error)
            print(f"\nTotal loss: {total_loss}\n")
            return total_loss

        except Exception as e:
            raise e

    def _create_dir_return_path(self, model_dir, filename):
        """
        Creates a directory and returns the file path.

        Parameters:
        - model_dir (str): The directory to create.
        - filename (str): The filename.

        Returns:
        - str: The file path.

        """
        try:
            os.makedirs(model_dir, exist_ok=True)
            return os.path.join(model_dir, filename)

        except Exception as e:
            raise e

    def save(self, filename, model_dir=None):
        """
        Saves the perceptron model to a file.

        Parameters:
        - filename (str): The filename.
        - model_dir (str): The directory to save the model (optional).

        """
        try:
            if model_dir is not None:
                model_file_path = self._create_dir_return_path(model_dir, filename)
                joblib.dump(self, model_file_path)

            else:
                model_file_path = self._create_dir_return_path('model', filename)
                joblib.dump(self, model_file_path)
        except Exception as e:
            raise e

    def load(self, filepath):
        """
        Loads a perceptron model from a file.

        Parameters:
        - filepath (str): The file path.

        Returns:
        - Perceptron: The loaded perceptron model.

        """
        try:
            return joblib.load(filepath)
        except Exception as e:
            raise e