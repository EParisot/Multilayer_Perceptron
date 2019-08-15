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
    def __init__(self, in_shape, width, activation, use_bias=True):
        self.act_name = activation
        self.activation = activations_dict[activation]
        self.deriv_act = derivatives_dict[activation]
        self.use_bias = use_bias
        if use_bias == True:
            self.width = width + 1
        else:
            self.width = width
        self.in_shape = in_shape
        self.weights = np.random.random_sample((self.width, self.in_shape))
        self.z = np.zeros((self.width,  self.in_shape))
        self.layer_out = np.zeros((self.width))
        self.delta = None
    
    def show(self):
        print("FullyConnected    : %s perceptrons, activation : %s" % (self.width, self.act_name))
        print(self.weights)
    
    def feedforward(self, X):
        pass