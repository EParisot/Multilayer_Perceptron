import numpy as np
from perceptrons import Perceptron

class Input(object):
    def __init__(self, in_shape, activation):
        self.features = in_shape[0]
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.tensor = np.zeros(self.features + 1)
        self.tensor[0] = 1.0
        self.weights = np.random.random_sample((self.features + 1,))
    
    def show(self):
        print("Input             : %s features,    activation : %s" % (self.features, self.act_name))
        print(self.tensor)
        print(self.weights)
    
    def feedforward(self, X):
        self.tensor = np.zeros(self.features + 1)
        self.tensor[0] = 1.0
        for i, elem in enumerate(X):
            self.tensor[1+i] = np.array(elem)
        scalar_prod = np.dot(self.tensor, self.weights)
        self.tensor = self.activation(scalar_prod)

class FC(object):
    def __init__(self, in_shape, width, activation):
        self.width = width
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.tensor = np.zeros(self.width + 1)
        self.tensor[0] = 1.0
        self.weights = np.random.random_sample((self.width + 1,))
    
    def show(self):
        print("FullyConnected    : %s perceptrons, activation : %s" % (self.width + 1, self.act_name))
        print(self.tensor)
        print(self.weights)
    
    def feedforward(self, X):
        self.tensor = np.zeros(self.width + 1)
        self.tensor[0] = 1.0
        for i in range(self.width):
            self.tensor[1+i] = X
        scalar_prod = np.dot(self.tensor, self.weights)
        self.tensor = self.activation(scalar_prod)

class Output(object):
    def __init__(self, in_shape, out_dim, activation):
        self.width = out_dim
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.tensor = np.zeros(self.width)
    
    def show(self):
        print("Output            : %s classes,     activation : %s" % (self.width, self.act_name))
        print(self.tensor)

    def feedforward(self, X):
        self.tensor = np.zeros(self.width)
        for i in range(self.width):
            self.tensor[i] = X
        self.tensor = self.activation(self.tensor)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return x if x > 0 else 0

activations_dict = {
  "sigmoid": sigmoid,
  "relu": relu
}