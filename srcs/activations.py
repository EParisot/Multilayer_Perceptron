import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
  return np.exp(x) / np.sum(np.exp(x))

activations_dict = {
  "sigmoid": sigmoid,
  "softmax": softmax
}

def sigmoid_deriv(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

derivatives_dict = {
  "sigmoid": sigmoid_deriv,
}
