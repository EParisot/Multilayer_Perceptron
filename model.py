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
        return layer.width
    
    def train(self, X, Y, batch_size=32, epochs=1, lr=0.1):
        for epoch in range(epochs):
            for batch_idx in range(0, X.shape[1], batch_size):
                batch = X[:, batch_idx : batch_idx + batch_size]
                batch_labels = Y[batch_idx : batch_idx + batch_size]
                # loop over batch
                for i, col in enumerate(batch.T):
                    # feedforward
                    for j, layer in enumerate(self.layers):
                        if isinstance(layer, Input):
                            layer.layer_out = np.zeros(layer.width)
                            layer.layer_out[0] = 1.
                            for k, val in enumerate(col):
                                layer.layer_out[k+1] = val
                        else:
                            layer.layer_in = self.layers[j - 1].layer_out
                            layer.z = np.dot(layer.layer_in, layer.weights.T)
                            layer.layer_out = layer.activation(layer.z)

                    # calc step error
                    loss_1 = np.multiply(batch_labels[i], np.log(layer.layer_out))
                    loss_2 = np.multiply((1 - batch_labels[i]), np.log(1 - layer.layer_out))
                    step_loss = -np.mean(loss_1 + loss_2)
                    print(step_loss)

                    # backprop
                    for layer in reversed(range(len(self.layers))):
                        if not isinstance(self.layers[layer], Input):
                            if self.layers[layer].is_last:
                                self.layers[layer].deltas = self.layers[layer].layer_out - batch_labels[i]
                            else:
                                deltas_agreg = np.dot(self.layers[layer+1].weights.T, self.layers[layer+1].deltas)
                                deltas_act = self.layers[layer].deriv_act(self.layers[layer].z)
                                self.layers[layer].deltas = np.multiply(deltas_agreg, deltas_act)

                    # compute gradients
                    for j, layer in enumerate(self.layers):
                        if not isinstance(layer, Input):
                            layer.gradients = np.outer(layer.deltas, self.layers[j-1].layer_out)

                    # update weights
                    for j, layer in enumerate(self.layers):
                        if not isinstance(layer, Input):
                            layer.weights -= lr * layer.gradients
