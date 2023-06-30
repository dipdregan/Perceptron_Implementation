from utils.all_utils import prepare_data,save_plot
from utils.models import Perceptron
import pandas as pd

def main(data, modelName, plotName, eta, epochs):

    df_OR = pd.DataFrame(data)
    x,y = prepare_data(df_OR)

    model_or = Perceptron(eta=eta,epochs=epochs)
    model_or.fit(x,y)
    
    _ = model_or.total_loss()

    # Saving the model
    model_or.save(filename=modelName,model_dir="Logical_model")

    # creating a plot and save
    save_plot(df_OR,model=model_or,plot_dir='Distribution_Plots',filename=plotName)


if __name__ == "__main__":

    OR = {
        'x1': [0,0,1,1],
        'x2': [0,1,0,1],
        'y' : [0,1,1,1]
    }

    ETA = 0.1 # eta lies between 0 to 1
    EPOCHS = 10
    main(data = OR, modelName = 'OR.model',plotName = 'OR.png',eta = ETA, epochs= EPOCHS)


