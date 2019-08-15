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
                # feedforward
                batch_loss = []
                batch = X[:, batch_idx : batch_idx + batch_size]
                labels_batch = Y[batch_idx : batch_idx + batch_size]
                # batch loop
                for j in range(len(batch[0])):
                    # layers loop
                    for i, layer in enumerate(self.layers):
                        if isinstance(layer, Input):
                            layer.feedforward(batch[:, j])
                        else:
                            layer.feedforward(self.layers[i - 1].layer_out)
                    # backprop
                    exit(0)
                    # calc step error
                    loss_1 = Y[batch_idx + j] * np.log(self.layers[-1].layer_out)
                    loss_2 = (1 - Y[batch_idx + j]) * np.log(1 - self.layers[-1].layer_out)
                    step_loss = -np.mean(loss_1 + loss_2)
                    batch_loss.append(step_loss)
                # calc batch error
                batch_loss = np.mean(batch_loss)
                print(epoch, batch_loss)