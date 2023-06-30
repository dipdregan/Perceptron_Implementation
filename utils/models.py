import numpy as np
import os
import joblib


class Perceptron:
    def __init__(self,eta: float=None, epochs: int=None): 
        #eta stand for learning rate
        #epochs =complete cycle of (forword_propogation+backword_propogation)
        try:
            self.weights = np.random.randn(3)*1e-4 # giving small random weights
            training = (eta is not None) and (epochs is not None)
            
            if training:
                print(f"Initial weights before training :\n {self.weights}")
            self.eta = eta
            self.epochs = epochs
            
        except Exception as e:
            raise e
            
    def _z_outcome(self, inputs_with_bais, weights):
        try:
            return np.dot(inputs_with_bais,weights)
        except Exception as e:
            raise e

    def activation_funtion(self,z):
        try:
            return np.where(z > 0 ,1,0)
        except Exception as e:
            raise e

    def fit(self, x, y):
        try:
            self.x = x
            self.y = y
            
            x_with_bais = np.c_[self.x,-np.ones((len(self.x), 1))]
            print(f"X with bais : \n{x_with_bais}")

            for epoch in range(self.epochs):
                print("--"*10)
                print(f"for epoch >> {epoch}")
                print("--"*10)
                
                z = self._z_outcome(x_with_bais, self.weights)
                y_hat = self.activation_funtion(z)
                print(f"Predicted value after forward pass : \n{y_hat}")
                
                self.error = self.y-y_hat
                print(f"error : \n{self.error}")
            
                self.weights = self.weights+ self.eta*np.dot(x_with_bais.T,self.error)
                print(f"Updated weights after epochs : {epoch+1}/{self.epochs} : \n{self.weights}")
                print("===="*10)
                
        except Exception as e:
            raise e
    
    def predict(self,x):
        try:
            x_with_bais = np.c_[x,-np.ones((len(x), 1))]
            z = self._z_outcome(x_with_bais,self.weights)
            return self.activation_funtion(z)
            
        except Exception as e:
            raise e
    
    def total_loss(self):
        try:
            total_loss = np.sum(self.error)
            print(f"\n total loss : {total_loss}\n")
            return total_loss
            
        except Exception as e:
            raise e
    
    def _create_dir_return_path(self, model_dir, filename):
        try:
            os.makedirs(model_dir,exist_ok= True)
            return os.path.join(model_dir,filename)
            
        except Exception as e:
            raise e
    
    def save(self,filename,model_dir=None):
        try:
            if model_dir is not None:
                model_file_path = self._create_dir_return_path(model_dir, filename)
                joblib.dump(self,model_file_path)
                
            else:
                model_file_path = self._create_dir_return_path('model',filename)
                joblib.dump(self,model_file_path)
        except Exception as e:
            raise e

    def load(self,filepath):
        try:
            return joblib.load(filepath)
        except Exception as e:
            raise e
    
        