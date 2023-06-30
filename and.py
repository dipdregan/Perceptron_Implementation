from utils.all_utils import prepare_data, save_plot
from utils.models import Perceptron
import pandas as pd

def main(data, modelName, plotName, eta, epochs):
    """
    Main function to train a Perceptron model and save both the model and the plot.

    Parameters:
    - data (dict): The dictionary containing input data with 'x1', 'x2', and 'y' keys.
    - modelName (str): The filename to save the trained model.
    - plotName (str): The filename to save the plot.
    - eta (float): The learning rate for the Perceptron.
    - epochs (int): The number of epochs (iterations) for training.

    Author: Dipendra Pratap Singh
    """
    try:
        df_AND = pd.DataFrame(data)
        x, y = prepare_data(df_AND)

        model_and = Perceptron(eta=eta, epochs=epochs)
        model_and.fit(x, y)

        _ = model_and.total_loss()

        # Saving the model
        model_and.save(filename=modelName, model_dir="Logical_model")

        # Creating a plot and saving
        save_plot(df_AND, model=model_and, plot_dir='Distribution_Plots', filename=plotName)

    except Exception as e:
        raise e

if __name__ == "__main__":
    """
    This code snippet runs when the script is executed as the main program.
    """
    AND = {
        'x1': [0, 0, 1, 1],
        'x2': [0, 1, 0, 1],
        'y': [0, 0, 0, 1]
    }

    ETA = 0.1  # eta lies between 0 to 1
    EPOCHS = 10
    main(data=AND, modelName='And.model', plotName='And.png', eta=ETA, epochs=EPOCHS)
