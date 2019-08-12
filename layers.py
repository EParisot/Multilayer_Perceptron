import numpy as np
from activations import activations_dict, gradients_dict
from perceptrons import Perceptron

class Input(object):
    def __init__(self, in_shape, activation):
        self.features = in_shape[0]
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.layer_in = np.zeros(self.features + 1)
        self.layer_in[0] = 1.0
        self.weights = np.random.random_sample((self.features + 1,))
        self.layer_out = np.zeros(self.features + 1)
    
    def show(self):
        print("Input             : %s features,    activation : %s" % (self.features, self.act_name))
        print(self.layer_in)
        print(self.weights)
    
    def feedforward(self, X):
        self.layer_in = np.zeros(self.features + 1)
        self.layer_in[0] = 1.0
        for i, elem in enumerate(X):
            self.layer_in[1+i] = elem
        self.layer_out = self.activation(np.dot(self.layer_in, self.weights))

    def gradient_descent(self, X, Y, lr):
        pass

class FC(object):
    def __init__(self, in_shape, width, activation):
        self.width = width
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.layer_in = np.zeros(self.width + 1)
        self.layer_in[0] = 1.0
        self.weights = np.random.random_sample((self.width + 1,))
        self.layer_out = None
    
    def show(self):
        print("FullyConnected    : %s perceptrons, activation : %s" % (self.width + 1, self.act_name))
        print(self.layer_in)
        print(self.weights)
    
    def feedforward(self, X):
        self.layer_in = np.zeros(self.width + 1)
        self.layer_in[0] = 1.0
        for i in range(self.width):
            self.layer_in[1+i] = X
        self.scalar_prod = self.activation(np.dot(self.layer_in, self.weights))
    
    def gradient_descent(self, prev_layer, Y, lr):
        pass

class Output(object):
    def __init__(self, in_shape, out_dim, activation):
        self.width = out_dim
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.layer_in = np.zeros(self.width)
        self.layer_out = None
    
    def show(self):
        print("Output            : %s classes,     activation : %s" % (self.width, self.act_name))
        print(self.layer_in)

    def feedforward(self, X):
        self.layer_in = np.zeros(self.width)
        for i in range(self.width):
            self.layer_in[i] = X
        self.layer_out = self.activation(self.layer_in)