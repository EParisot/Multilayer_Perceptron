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
                labels_batch = Y[batch_idx : batch_idx + batch_size]
                # batch loop
                for i, layer in enumerate(self.layers):
                    # feedforward
                    if isinstance(layer, Input):
                        layer.feedforward(batch.T)
                    else:
                        layer.feedforward(self.layers[i - 1].layer_out)
                
                # calc step error
                loss_1 = np.multiply(labels_batch, np.log(self.layers[-1].layer_out))
                loss_2 = np.multiply((1 - labels_batch), np.log(1 - self.layers[-1].layer_out))
                batch_loss = -np.mean(loss_1 + loss_2)
                print(batch_loss, labels_batch, layer.layer_out)
                                    
                # backprop
                for layer in reversed(range(len(self.layers))):
                    if not isinstance(self.layers[layer], Input):
                        if self.layers[layer].is_last:
                            self.layers[layer].backprop(labels_batch)
                        else:
                            self.layers[layer].backprop(self.layers[layer + 1])