import os 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import numpy as np

def prepare_data(df, target_col='y'):
    """
    Prepares the data for training by separating the features and target variable.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - target_col (str): The column name of the target variable.

    Returns:
    - tuple: A tuple containing the features (X) and the target variable (y).

    """
    try:
        x = df.drop(target_col,axis=1)
        y = df[target_col]
        return x, y

    except Exception as e:
        raise e 

def save_plot(df, model, plot_dir='plots', filename='plot.png'):
    """
    Saves a scatter plot with decision regions based on the given model and data.

    Parameters:
    - df (DataFrame): The input DataFrame containing the data.
    - model: The trained model object with a predict() method.
    - plot_dir (str): The directory to save the plot.
    - filename (str): The filename of the plot.

    """
    try:
        def _create_base_plot(df):
            """
            Creates the base scatter plot with data points.

            Parameters:
            - df (DataFrame): The input DataFrame containing the data.

            """
            df.plot(kind='scatter', x='x1', y='x2', c='y', s=100, cmap='coolwarm')
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
            plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
            figure = plt.gcf()
            figure.set_size_inches(10, 8)

        def _plot_decision_regions(X, y, classifier, resolution=0.02):
            """
            Plots the decision regions based on the given classifier.

            Parameters:
            - X (DataFrame): The input features.
            - y (Series): The target variable.
            - classifier: The trained classifier object with a predict() method.
            - resolution (float): The resolution of the decision regions.

            """
            colors = ('cyan', 'lightgreen')
            cmap = ListedColormap(colors)
            X = X.values
            x1 = X[:, 0]
            x2 = X[:, 1]
            x1_min, x1_max = x1.min() - 1, x1.max() + 1
            x2_min, x2_max = x2.min() - 1, x2.max() + 1 
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))
            y_hat = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
            y_hat = y_hat.reshape(xx1.shape)
            plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
            plt.xlim(xx1.min(), xx1.max())
            plt.xlim(xx2.min(), xx2.max())
            plt.plot()

        X, y = prepare_data(df)
        _create_base_plot(df)
        _plot_decision_regions(X, y, model)

        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, filename)
        plt.savefig(plot_path)

    except Exception as e:
        raise e
