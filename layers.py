import numpy as np
from activations import activations_dict, derivatives_dict

class Input(object):
    def __init__(self, in_shape):
        self.width = in_shape[0] + 1
        self.layer_out = np.zeros(self.width)
    
    def show(self):
        print("Input             : %s features" % (self.width))
    
    def feedforward(self, X):
        self.layer_out = np.zeros((len(X), self.width))
        self.layer_out[:, 0] = np.ones(len(X))
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                self.layer_out[row][col+1] = X[row][col]

class FC(object):
    def __init__(self, in_shape, width, activation, is_last=False):
        self.act_name = activation
        self.activation = activations_dict[activation]
        #self.deriv_act = derivatives_dict[activation]
        self.is_last = is_last
        if is_last == False:
            self.width = width + 1
        else:
            self.width = width
        self.in_shape = in_shape
        self.weights = np.random.random_sample((self.width, self.in_shape))
        self.z = np.zeros((self.width,))
        self.layer_out = np.zeros((self.width,))
        self.deltas = np.zeros((self.width,))
    
    def show(self):
        print("FullyConnected    : %s perceptrons, activation : %s" % (self.width, self.act_name))
        print(self.in_shape)
        print(self.weights.shape)
        print(self.deltas.shape)
        print(self.layer_out.shape)
    
    def feedforward(self, X):
        self.layer_in = X
        self.z = np.dot(self.layer_in, self.weights.T)
        self.layer_out = self.activation(self.z)
    
    def backprop(self, Y):
        if self.is_last == True:
            pass
        else:
            pass
        


        