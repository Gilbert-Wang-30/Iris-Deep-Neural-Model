import numpy as np
import math
from data_loader import load_data


def initialize_data(*args):
    """
    Prepares data for training, validation, and testing.
    
    Args:
    *args: Optional arguments. 
        - If no arguments, returns train, dev, and test sets.
        - If two arguments are provided (number of classes and mini-batch size), 
          returns mini-batches along with the datasets.

    Returns:
    - If no args: train_features, train_labels, dev_features, dev_labels, test_features, test_labels
    - If args: mini_batches, train_features, train_labels, dev_features, dev_labels, test_features, test_labels
    """
    # Load the data
    (train_features_orig, train_labels_orig), (dev_features_orig, dev_labels_orig), (test_features_orig, test_labels_orig) = load_data()
    
    # Get dataset sizes
    m_train = train_features_orig.shape[1]
    m_dev = dev_features_orig.shape[1]
    m_test = test_features_orig.shape[1]

    # Reshape labels for compatibility
    train_labels = train_labels_orig.reshape(1, m_train)
    dev_labels = dev_labels_orig.reshape(1, m_dev)
    test_labels = test_labels_orig.reshape(1, m_test)

    # Case 1: If no extra arguments passed, return the data without mini-batches
    if len(args) == 0:
        return train_features_orig, train_labels, dev_features_orig, dev_labels, test_features_orig, test_labels

    # Case 2: If two arguments passed (number of classes and mini-batch size), return mini-batches
    elif len(args) == 2:
        num_classes, mini_batch_size = args
        mini_batches = random_mini_batches(train_features_orig, one_hot(train_labels, num_classes), mini_batch_size, seed=0)
        return mini_batches, train_features_orig, train_labels, dev_features_orig, dev_labels, test_features_orig, test_labels

    # Raise an error if more arguments than expected are passed
    else:
        raise ValueError("Too many arguments passed to initialize_data")


def random_mini_batches(X, Y, mini_batch_size = 32, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    """These lines Shuffle train set, not needed since already shuffled
    permutation = list(np.random.permutation(m))
    X = X[:, permutation]
    Y = Y[:, permutation].reshape((1, m))
    """
    # Step 2 - Partition (shuffled_X, shuffled_Y).
    # Cases with a complete mini batch size only i.e each of 32 examples.
    num_complete_minibatches = math.floor(m / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        
        mini_batch_X = X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # For handling the end case (last mini-batch < mini_batch_size i.e less than 64)
    if m % mini_batch_size != 0:
        
        mini_batch_X = X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = Y[:, num_complete_minibatches * mini_batch_size : m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def initialize_parameters(layer_dims, initialization):
    
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        if initialization[l-1] == 'he':
            parameters['W' + str(l)], parameters['b' + str(l)] = he_initialization_parameters(layer_dims[l-1], layer_dims[l])
        elif initialization[l-1] == 'xavier':
            parameters['W' + str(l)], parameters['b' + str(l)] = xavier_initialization_parameters(layer_dims[l-1], layer_dims[l])
        else:
            raise ValueError(f"Initialization method {initialization} not supported")
        
    return parameters

def he_initialization_parameters(n_x, n_z):
    W = np.random.randn(n_z, n_x) * np.sqrt(2 / n_x)
    b = np.zeros((n_z, 1))
    return W, b

def xavier_initialization_parameters(n_x, n_z):
    limit = np.sqrt(6 / (n_x + n_z))
    # Initialize weights with uniform distribution within [-limit, limit]
    W = np.random.uniform(low=-limit, high=limit, size=(n_z, n_x))
    b = np.zeros((n_z, 1))
    return W, b

def one_hot(Y, C):
    Matrix = np.zeros((C, Y.shape[1]))
    for c in range(Y.shape[1]):
        Matrix[Y[0, c], c] = 1

    return Matrix