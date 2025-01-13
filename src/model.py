import numpy as np
import math
import h5py
import struct
from data_loader import load_data
from activations import *
from initialize import *


def nn_model(mini_batches, learning_rate, lambd, decay_rate, parameters, adam_velocity, adam_rms, t, layer_dims= [784, 256, 128, 10], activications= ['leaky_relu', 'leaky_relu', 'softmax'], print_cost=False):
    """
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- one_hot label" vector 
    layer_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    activications -- list containing the activation functions for each layer
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 1000 iterations
    
    Returns: trained parameters after one epoch
    """
    
    # Initialize parameters
    
    m = mini_batches[1][1].shape[1]
    L = len(layer_dims)
    num_minibatches = len(mini_batches)
    costs = []
    total_cost = 0
    # Test all the parameters
    #print(f"Parameters W1: {parameters['W1']}, m: {m}, L: {L}")
    t+=1
    # Loop (gradient descent of each mini-batch)
    for i in range(0, num_minibatches):
        # Forward propagation
        A, cost, cache = forward_propagation(mini_batches[i][0], mini_batches[i][1], lambd, parameters, activications)
        # Backward propagation
        grads = backward_propagation(cache, mini_batches[i][1], parameters, activications, lambd)
        # Update parameters
        total_cost += cost*mini_batches[i][0].shape[1]
        parameters, adam_velocity, adam_rms = adam_optimize_parameters(adam_velocity, adam_rms, parameters, grads, t, learning_rate, decay_rate)
        # Print the cost every 1000 iterations
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
        costs.append(cost)
    return parameters, costs, total_cost, t, adam_velocity, adam_rms


    #return parameters

def forward_propagation(X, Y, lambd, parameters, activations):
    A = X
    L = len(activations)+1
    cache={}
    cache['A0'] = X

    for i in range(1, L):
        A_prev = A
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]

        A, Z = forward_propagation_layer(A_prev, W, b, activations[i-1])
        cache['A'+str(i)] = A
        cache['Z'+str(i)] = Z
    y_hat = A
    cost = compute_cost_L2(y_hat, Y, parameters, lambd)
    return y_hat, cost, cache



def forward_propagation_layer(X, W, b, activation):
    Z = np.dot(W, X) + b
    if(activation== 'leaky_relu'):
        A = leaky_relu(Z)
    elif(activation== 'relu'):
        A = relu(Z)
    elif(activation== 'sigmoid'):
        A = sigmoid(Z)
    elif(activation== 'softmax'):
        A = softmax(Z)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    
    return A, Z

def compute_cost_L2(y_hat, Y, parameters, lambd):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (10, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), shape (10, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    L = len(parameters)//2
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10) 
    cross_entropy_cost = -np.sum(Y * np.log(y_hat)) / m
    w_sum = 0
    for i in range(1, L):
        w_sum += np.sum(np.square(parameters['W'+str(i)]))
    
    L2_regularization_cost = (lambd/(2*m)) * w_sum
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

def backward_propagation(cache, Y, parameters, activation, lambd):
    grads = {}
    L = len(activation)
    dA_prev, dW, db = softmax_backward(cache['A'+str(L)], Y, cache['A'+str(L-1)], parameters['W'+str(L)], lambd)
    grads['dW'+str(L)] = dW
    grads['db'+str(L)] = db
    for i in range(L-1, 0, -1):
        dA_prev, dW, db = backward_propagation_layer(dA_prev, cache['Z'+str(i)], cache['A'+str(i-1)], parameters['W'+str(i)], activation[i-1])
        grads['dW'+str(i)] = dW
        grads['db'+str(i)] = db
    return grads

def softmax_backward(y_hat, Y, A_prev, W, lambd):
    m = A_prev.shape[1]
    dZ = y_hat - Y
    dW = np.dot(dZ, A_prev.T) / m + (lambd/m) * W
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def backward_propagation_layer(dA, Z, A_prev, W, activation):
    m = A_prev.shape[1]
    if(activation== 'leaky_relu'):
        dZ = leaky_relu_backward(dA, Z)
    elif(activation== 'relu'):
        dZ = relu_backward(dA, Z)
    elif(activation== 'sigmoid'):
        dZ = sigmoid_backward(dA, Z)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")
    
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def leaky_relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] *= 0.01
    return dZ

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, Z):
    A = sigmoid(Z)
    dZ = dA * A * (1 - A)
    return dZ

def initialize_adam_velocity(parameters):
    L = len(parameters) // 2
    velocity = {}
    rms_velocity = {}

    for l in range(1, L + 1):
        velocity['dW' + str(l)] = np.zeros_like(parameters['W' + str(l)])
        velocity['db' + str(l)] = np.zeros_like(parameters['b' + str(l)])
        rms_velocity['dW' + str(l)] = np.zeros_like(parameters['W' + str(l)])
        rms_velocity['db' + str(l)] = np.zeros_like(parameters['b' + str(l)])

    return velocity, rms_velocity

def adam_optimize_parameters(velocity, rms_velocity, parameters, grads, t, learning_rate, decay_rate=0, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    L = len(parameters)//2
    # Apply time-based learning rate decay
    adjusted_learning_rate = learning_rate / (1 + decay_rate* t)

    
    for i in range(1, L+1):
        velocity['dW'+str(i)] = beta1 * velocity['dW'+str(i)] + (1 - beta1) * grads['dW'+str(i)]
        rms_velocity['dW'+str(i)] = beta2 * rms_velocity['dW'+str(i)] + (1 - beta2) * np.square(grads['dW'+str(i)])
        v_corrected = velocity['dW'+str(i)] / (1 - beta1**t)
        s_corrected = rms_velocity['dW'+str(i)] / (1 - beta2**t)
        parameters['W'+str(i)] -= adjusted_learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)
        
        velocity['db'+str(i)] = beta1 * velocity['db'+str(i)] + (1 - beta1) * grads['db'+str(i)]
        rms_velocity['db'+str(i)] = beta2 * rms_velocity['db'+str(i)] + (1 - beta2) * np.square(grads['db'+str(i)])
        v_corrected = velocity['db'+str(i)] / (1 - beta1**t)
        s_corrected = rms_velocity['db'+str(i)] / (1 - beta2**t)
        parameters['b'+str(i)] -= adjusted_learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return parameters, velocity, rms_velocity

#print(f"Training data: {test_features.shape}, Labels: {test_labels.shape}")