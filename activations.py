import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leakyRelu(x):
    return np.maximum(0.1 * x, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
  ret = []
  for elem in x:
    ret.append(np.exp(elem) / np.sum(np.exp(elem)))
  return np.array(ret)

activations_dict = {
  "sigmoid": sigmoid,
  "relu": relu,
  "leakyRelu": leakyRelu,
  "tanh": tanh,
  "softmax": softmax
}


def sigmoid_deriv(x):
    dx = x * (1 - x)
    return dx

def relu_deriv(x):
    dx = np.multiply(x, np.int64(x > 0))
    return dx

def leakyRelu_deriv(x):
    dx = np.multiply(x, np.int64(x > 0.1))
    return dx

def tanh_deriv(x):
    dx = x * (1 - np.square(x))
    return dx

derivatives_dict = {
  "sigmoid": sigmoid_deriv,
  "relu": relu_deriv,
  "leakyRelu": leakyRelu_deriv,
  "tanh": tanh_deriv
}
