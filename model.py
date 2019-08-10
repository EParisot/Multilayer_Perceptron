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
            for batch in range(0, X.shape[1], batch_size):
                for data in X[:, batch : batch + batch_size]:
                    for elem in data:
                        pass#print(elem)

