import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def leaky_relu(Z):
    return np.maximum(0.01 * Z, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

def activation_function(Z, activation):
    if activation == 'relu':
        return relu(Z)
    elif activation == 'sigmoid':
        return sigmoid(Z)
    elif activation == 'leaky_relu':
        return leaky_relu(Z)
    elif activation == 'softmax':
        return softmax(Z)
    else:
        raise ValueError(f"Activation function {activation} not supported")

