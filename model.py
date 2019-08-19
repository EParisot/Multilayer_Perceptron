from layers import Input
import matplotlib.pyplot as plt
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
    
    def train(self, X, Y, batch_size=32, epochs=1, lr=0.1, verbose=True):
        history = np.zeros((epochs, 2))
        for epoch in range(epochs):
            # split batches
            for batch_start in range(0, len(X), batch_size):
                batch_data = X[batch_start : batch_start + batch_size]
                batch_labels = Y[batch_start : batch_start + batch_size]
                # init gradients
                for layer in self.layers[1:]:
                    layer.w_gradients = np.zeros((layer.weights.shape[0], layer.weights.shape[1]))
                    layer.b_gradients = np.zeros((layer.biases.shape[0]))
                # loop over batch
                for i, row in enumerate(batch_data):
                    # feedforward
                    self.feedforward(row)
                    # backprop
                    self.backprop(batch_labels[i])
                    # compute gradients
                    self.gradients()
                # update weights
                self.update_weights(lr, len(batch_data))
            loss, acc = self.evaluate(X, Y)
            if verbose == True:
                print("Epoch %s, loss : %0.2f, acc : %0.2f" % (epoch, loss, acc*100))
                history[epoch] = (loss, acc)
        if verbose == True:
            plt.figure("Train history")
            plt.plot(history[:, 0], label="loss")
            plt.plot(history[:, 1], label="acc")
            plt.legend()
            plt.show()

    def feedforward(self, data):
        for j, layer in enumerate(self.layers):
            if isinstance(layer, Input):
                layer.layer_out = data
            else:
                layer.layer_in = self.layers[j - 1].layer_out
                layer.z = np.dot(layer.weights, layer.layer_in) + layer.biases
                layer.layer_out = layer.activation(layer.z)
        return layer.layer_out

    def backprop(self, label):
        for layer in reversed(range(len(self.layers))):
            if not isinstance(self.layers[layer], Input):
                if self.layers[layer].is_last:
                    self.layers[layer].deltas = self.layers[layer].layer_out - label
                else:
                    deltas_act = self.layers[layer].deriv_act(self.layers[layer].z)
                    deltas_agreg = np.dot(self.layers[layer+1].weights.T, self.layers[layer+1].deltas)
                    self.layers[layer].deltas = np.multiply(deltas_act, deltas_agreg)
    
    def gradients(self):
        for j, layer in enumerate(self.layers):
            if not isinstance(layer, Input):
                layer.w_gradients += np.outer(layer.deltas, self.layers[j-1].layer_out)
                layer.b_gradients += layer.deltas

    def update_weights(self, lr, batch_len):
        for layer in self.layers[1:]:
            avg_w_gradients = np.divide(layer.w_gradients, batch_len)
            avg_b_gradients = np.divide(layer.b_gradients, batch_len)
            layer.weights -= lr * avg_w_gradients
            layer.biases -= lr * avg_b_gradients
            
    
    def evaluate(self, X, Y):
        loss = []
        acc = []
        for i, val in enumerate(X):
            pred = self.feedforward(val)
            loss.append(self.step_error(pred, Y[i]))
            if np.argmax(pred) == np.argmax(Y[i]):
                acc.append(1)
            else:
                acc.append(0)
        return np.mean(loss), np.mean(acc)

    def step_error(self, pred, label):
        loss_1 = np.multiply(label, np.log(pred))
        loss_2 = np.multiply((1 - label), np.log(1 - pred))
        step_loss = -np.mean(loss_1 + loss_2)
        return step_loss