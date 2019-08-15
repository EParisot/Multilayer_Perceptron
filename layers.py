import numpy as np
from activations import activations_dict, derivatives_dict

class Input(object):
    def __init__(self, in_shape):
        self.width = in_shape[0] + 1
        self.layer_out = np.zeros(self.width)
    
    def show(self):
        print("Input             : %s features" % (self.width))
    
    def feedforward(self, X):
        self.layer_out = np.zeros(self.width)
        self.layer_out[0] = 1.0
        for i, elem in enumerate(X):
            self.layer_out[1+i] = elem

class FC(object):
    def __init__(self, in_shape, width, activation, is_last=False):
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.deriv_act = derivatives_dict[activation]
        self.is_last = is_last
        if is_last == False:
            self.width = width + 1
        else:
            self.width = width
        self.in_shape = in_shape
        self.weights = np.random.random_sample((self.width, self.in_shape))
        self.z = np.zeros((self.width,))
        self.layer_out = np.zeros((self.width,))
        self.deltas = np.zeros((self.width, self.in_shape))
    
    def show(self):
        print("FullyConnected    : %s perceptrons, activation : %s" % (self.width, self.act_name))
        print(self.in_shape)
        print(self.weights.shape)
        print(self.layer_out.shape)
    
    def feedforward(self, X):
        self.layer_in = X
        for neuron in range(self.width):
            self.z[neuron] = np.dot(X, self.weights[neuron])
            self.layer_out[neuron] = self.activation(self.z[neuron])
    
    def backprop(self, prev_layer, deltas, lr):
        for neuron in range(self.width):
            self.deltas = np.dot(deltas, self.weights.T[neuron]) * self.deriv_act(prev_layer.layer_out)
            #self.weights[neuron] -= lr * self.deltas * self.layer_in