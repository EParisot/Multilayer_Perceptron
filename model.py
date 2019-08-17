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
                # batch loop
                for i, layer in enumerate(self.layers):
                    # feedforward
                    if isinstance(layer, Input):
                        layer.feedforward(batch.T)
                    else:
                        layer.feedforward(self.layers[i - 1].layer_out)
                
                # calc step error
                loss_1 = np.multiply(batch_labels, np.log(layer.layer_out))
                loss_2 = np.multiply((1 - batch_labels), np.log(1 - layer.layer_out))
                batch_loss = -np.mean(loss_1 + loss_2)
                print(batch_loss, batch_labels, layer.layer_out)
                                    
                # backprop
                for layer in reversed(range(len(self.layers))):
                    if not isinstance(self.layers[layer], Input):
                        if self.layers[layer].is_last:
                            self.layers[layer].backprop(batch_labels)
                        else:
                            self.layers[layer].backprop(self.layers[layer + 1])