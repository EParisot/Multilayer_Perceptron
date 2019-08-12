import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leakyRelu(x):
    return np.maximum(0.1 * x, x)

def tanh(x):
    return np.tanh(x)

activations_dict = {
  "sigmoid": sigmoid,
  "relu": relu,
  "leakyRelu": leakyRelu,
  "tanh": tanh
}


def sigmoid_gradient(dy, x):
    y = sigmoid(x)
    dx = dy * y * (1 - y)
    return dx

def relu_gradient(dy, x):
    y = relu(x)
    dx = np.multiply(dy, np.int64(y > 0))
    return dx

def leakyRelu_gradient(dy, x):
    y = leakyRelu(x)
    dx = np.multiply(dy, np.int64(y > 0.1))
    return dx

def tanh_gradient(dy, x):
    y = tanh(x)
    dx = dy * (1 - np.square(y))
    return dx

gradients_dict = {
  "sigmoid": sigmoid_gradient,
  "relu": relu_gradient,
  "leakyRelu": leakyRelu_gradient,
  "tanh": tanh_gradient
}
