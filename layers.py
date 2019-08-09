from math import exp
from perceptrons import Perceptron

class Input(object):
    def __init__(self, shape):
        self.width = shape[-1]

class Output(object):
    def __init__(self, in_shape, out_dim, activation):
        self.width = out_dim
        self.perceptrons = [Perceptron(in_shape)]
        self.activation = activations_dict[activation]

class FC(object):
    def __init__(self, in_shape, width, activation):
        self.width = width
        self.perceptrons = [Perceptron(in_shape)]
        self.activation = activations_dict[activation]

def sigmoid(x):
    return 1 / (1 + exp(-x))

def relu(x):
    return x if x > 0 else 0

activations_dict = {
  "sigmoid": sigmoid,
  "relu": relu
}