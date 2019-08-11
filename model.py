from layers import Input

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
        for epoch in range(epochs):
            for batch_idx in range(0, X.shape[1], batch_size):
                # feedforward
                batch = X[:, batch_idx : batch_idx + batch_size]
                for j in range(len(batch[0])):
                    for i, layer in enumerate(self.layers):
                        if isinstance(layer, Input):
                            layer.feedforward(batch[:, j])
                        else:
                            layer.feedforward(self.layers[i - 1].tensor)
            print(self.layers[-1].tensor)
                # backpropagation
                
