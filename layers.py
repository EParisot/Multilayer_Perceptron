import numpy as np
from activations import activations_dict, derivatives_dict

class Input(object):
    def __init__(self, in_shape, activation):
        self.width = in_shape[0] + 1
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.deriv_act = derivatives_dict[activation]
        self.weights = np.random.random_sample((self.width,))
        self.layer_in = np.zeros(self.width)
        self.z = np.zeros(self.width)
        self.layer_out = np.zeros(self.width)
    
    def show(self):
        print("Input             : %s features,    activation : %s" % (self.width, self.act_name))
        print(self.weights)
    
    def feedforward(self, X):
        self.layer_in = np.zeros(self.width)
        self.layer_in[0] = 1.0
        for i, elem in enumerate(X):
            self.layer_in[1+i] = elem
        self.z = np.dot(self.layer_in, self.weights)
        self.layer_out = self.activation(self.z)

    def gradient(self, Y, Y_hat, lr):
        self.err = (self.layer_in * self.weights) * self.deriv_act(self.layer_out)
        self.weights -= lr * self.err * self.layer_in

class FC(object):
    def __init__(self, in_shape, width, activation):
        self.width = width + 1
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.deriv_act = derivatives_dict[activation]
        self.weights = np.random.random_sample((self.width,))
        self.layer_in = np.zeros(self.width)
        self.z = np.zeros(self.width)
        self.layer_out = np.zeros(self.width)
    
    def show(self):
        print("FullyConnected    : %s perceptrons, activation : %s" % (self.width, self.act_name))
        print(self.weights)
    
    def feedforward(self, X):
        self.layer_in = np.zeros(self.width)
        self.layer_in[0] = 1.0
        for i in range(self.width - 1):
            self.layer_in[1+i] = X
        self.z = np.dot(self.layer_in, self.weights)
        self.layer_out = self.activation(self.z)
    
    def gradient(self, Y_hat, Y, lr):
        self.err = (self.layer_in * self.weights) * self.deriv_act(self.layer_out)
        self.weights -= lr * self.err * self.layer_in

class Output(object):
    def __init__(self, in_shape, out_dim, activation):
        self.width = out_dim
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.deriv_act = derivatives_dict[activation]
        self.layer_in = np.zeros(self.width)
        self.layer_out = np.zeros(self.width)
    
    def show(self):
        print("Output            : %s classes,     activation : %s" % (self.width, self.act_name))
        print(self.layer_in)

    def feedforward(self, X):
        self.layer_in = np.zeros(self.width)
        for i in range(self.width):
            self.layer_in[i] = X
        self.layer_out = self.activation(self.layer_in)
