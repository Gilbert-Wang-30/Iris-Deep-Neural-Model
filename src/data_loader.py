import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

# This file is used to load the MNIST dataset from the ubyte files
# Is file is copied from online resources and is not written by me



def save_model(parameters, filename="model2.h5"):
    with h5py.File(filename, 'w') as f:
        for key, value in parameters.items():
            f.create_dataset(key, data=value)
    print(f"Model saved to {filename}")

def load_model(filename="model2.h5"):
    parameters = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            parameters[key] = np.array(f[key])
    print(f"Model loaded from {filename}")
    return parameters

def load_data():
    """
    Load the Iris dataset and split it into train, dev, and test sets.

    Returns:
    - (train_features, train_labels): Training features and labels
    - (dev_features, dev_labels): Development/validation features and labels
    - (test_features, test_labels): Testing features and labels
    """
    # Load the Iris dataset
    df = pd.read_csv("data/iris.csv")
    
    # Extract features and labels
    features = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
    labels = df['variety'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}).values

    # Split into training (80%), and temporary set (20%)
    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features, labels, test_size=0.3, random_state=11
    )

    # Further split the temporary set into dev (50%) and test (50%)
    dev_features, test_features, dev_labels, test_labels = train_test_split(
        temp_features, temp_labels, test_size=0.5, random_state=13
    )

    # Return data in expected format
    return (
        (train_features.T, train_labels),
        (dev_features.T, dev_labels),
        (test_features.T, test_labels)
    )