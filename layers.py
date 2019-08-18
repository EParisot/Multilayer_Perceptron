import numpy as np
from activations import activations_dict, derivatives_dict

class Input(object):
    def __init__(self, in_shape):
        self.width = in_shape[0] + 1
        self.layer_out = np.zeros(self.width)
    
    def show(self):
        print("Input             : %s features" % (self.width))
    
class FC(object):
    def __init__(self, in_shape, width, activation, is_last=False):
        self.act_name = activation
        self.activation = activations_dict[activation]
        if self.act_name != "softmax":
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
    
    def show(self):
        print("FullyConnected    : %s perceptrons, activation : %s" % (self.width, self.act_name))
        print(self.in_shape)
        print(self.weights.shape)
        print(self.layer_out.shape)