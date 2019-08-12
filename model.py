from layers import Input, Output
import numpy as np

class Model(object):
    def __init__(self):
        self.layers = []
    
    def show(self):
        for layer in self.layers:
            layer.show()

    def add(self, layer):
        self.layers.append(layer)
        return layer.layer_in.shape[0]
    
    def train(self, X, Y, batch_size=32, epochs=1, lr=0.1):
        for _ in range(epochs):
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
                    # calc step error
                    loss_1 = Y[batch_idx + j] * np.log(self.layers[-1].layer_out)
                    loss_2 = (1 - Y[batch_idx + j]) * np.log(1 - self.layers[-1].layer_out)
                    step_loss = -np.mean(loss_1 + loss_2)
                    batch_loss.append(step_loss)
                # calc batch error
                batch_loss = np.mean(batch_loss)
                
                # backpropagation
                for i, layer in enumerate(reversed(self.layers)):
                    if not isinstance(layer, Output):
                        layer.gradient_descent(self.layers[i - 1], labels_batch, lr)
                        
