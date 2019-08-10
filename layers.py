import numpy as np
from perceptrons import Perceptron

class Input(object):
    def __init__(self, shape):
        self.width = shape[0]
        self.tensor = np.zeros(self.width + 1)
    def show(self):
        print("Input            : %s features" % self.width)

class Output(object):
    def __init__(self, in_shape, out_dim, activation):
        self.width = out_dim
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.tensor = np.zeros(self.width)
    def show(self):
        print("Output           : %s classes,     activation : %s" % (self.width, self.act_name))

class FC(object):
    def __init__(self, in_shape, width, activation):
        self.width = width
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.tensor = np.zeros(self.width + 1)
    def show(self):
        print("FullyConnected   : %s perceptrons, activation : %s" % (self.width, self.act_name))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return x if x > 0 else 0

activations_dict = {
  "sigmoid": sigmoid,
  "relu": relu
}