import numpy as np

from activations import activations_dict, derivatives_dict

#np.random.seed(42)

class Input(object):
    def __init__(self, in_shape):
        self.width = in_shape[1]
        self.layer_out = np.zeros(self.width)
    
    def show(self):
        print("Input             : %s features \n" % (self.width))
    
class FC(object):
    def __init__(self, in_shape, width, activation, is_last=False):
        self.in_shape = in_shape
        self.is_last = is_last
        self.width = width
        self.act_name = activation
        self.activation = activations_dict[activation]
        if self.act_name != "softmax":
            self.deriv_act = derivatives_dict[activation]
        self.weights = np.random.randn(self.width, self.in_shape)
        self.biases = np.random.randn(self.width)
        self.z = np.zeros(self.width)
        self.layer_out = np.zeros(self.width)
    
    def show(self):
        print("FullyConnected    : %s perceptrons, activation : %s" % (self.width, self.act_name))
        print("in_shape : %s, weights : %s, out_shape : %s \n" % (self.in_shape, self.weights.shape, self.layer_out.shape[0]))