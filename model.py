from layers import Input
import numpy as np

class Model(object):
    def __init__(self):
        self.layers = []
    
    def show(self):
        for layer in self.layers:
            layer.show()

    def add(self, layer):
        self.layers.append(layer)
        return layer.tensor.shape[0]
    
    def train(self, X, Y, batch_size=32, epochs=1):
        for _ in range(epochs):
            for batch_idx in range(0, X.shape[1], batch_size):
                # feedforward
                batch_err = []
                batch = X[:, batch_idx : batch_idx + batch_size]
                for j in range(len(batch[0])):
                    for i, layer in enumerate(self.layers):
                        if isinstance(layer, Input):
                            layer.feedforward(batch[:, j])
                        else:
                            layer.feedforward(self.layers[i - 1].tensor)
                    # calc error
                    sq_err_1 = Y[batch_idx + j] * np.log(self.layers[-1].tensor)
                    sq_err_2 = (1 - Y[batch_idx + j]) * np.log(1 - self.layers[-1].tensor)
                    step_err = -np.mean(sq_err_1 + sq_err_2)
                    batch_err.append(step_err)
                batch_err = np.sum(batch_err)
                # backpropagation
                
