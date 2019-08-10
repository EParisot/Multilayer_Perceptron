from layers import Input

class Model(object):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        return layer.tensor.shape[0]