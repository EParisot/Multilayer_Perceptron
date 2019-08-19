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
                batch = X[batch_idx : batch_idx + batch_size]
                batch_labels = Y[batch_idx : batch_idx + batch_size]
                for layer in self.layers:
                    layer.w_gradients = []
                    layer.b_gradients = []
                batch_loss = []
                # loop over batch
                for i, row in enumerate(batch):
                    # feedforward
                    self.feedforward(row)
                    # calc step error
                    batch_loss.append(self.step_error(layer, batch_labels[i]))
                    # backprop
                    self.backprop(batch_labels[i])
                    # compute gradients
                    self.gradients()
                # update weights
                self.update_weights(lr)
                print(epoch, np.mean(batch_loss))

    def feedforward(self, data):
        for j, layer in enumerate(self.layers):
            if isinstance(layer, Input):
                layer.layer_out = data
            else:
                layer.layer_in = self.layers[j - 1].layer_out
                layer.z = np.dot(layer.weights, layer.layer_in) + layer.biases
                layer.layer_out = layer.activation(layer.z)

    def step_error(self, last_layer, label):
        loss_1 = np.multiply(label, np.log(last_layer.layer_out))
        loss_2 = np.multiply((1 - label), np.log(1 - last_layer.layer_out))
        step_loss = -np.mean(loss_1 + loss_2)
        print(last_layer.layer_out, label)
        return step_loss

    def backprop(self, label):
        for layer in reversed(range(len(self.layers))):
            if not isinstance(self.layers[layer], Input):
                if self.layers[layer].is_last:
                    self.layers[layer].deltas = self.layers[layer].layer_out - label
                else:
                    deltas_agreg = np.dot(self.layers[layer+1].weights.T, self.layers[layer+1].deltas)
                    deltas_act = self.layers[layer].deriv_act(self.layers[layer].z)
                    self.layers[layer].deltas = np.multiply(deltas_agreg, deltas_act)
    
    def gradients(self):
        for j, layer in enumerate(self.layers):
            if not isinstance(layer, Input):
                layer.w_gradients.append(np.outer(layer.deltas, self.layers[j-1].layer_out))
                layer.b_gradients.append(layer.deltas)

    def update_weights(self, lr):
        for layer in self.layers:
            if not isinstance(layer, Input):
                layer.weights -= lr * np.mean(layer.w_gradients)
                layer.biases -= lr * np.mean(layer.b_gradients)