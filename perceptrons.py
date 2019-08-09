class Perceptron(object):
    def __init__(self, in_shape):
        self.in_shape = in_shape
        self.weights = [0.0 for elem in range(in_shape + 1)]
