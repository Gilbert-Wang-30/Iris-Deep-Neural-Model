import numpy as np
from data_loader import load_data
from initialize import initialize_data
from model import forward_propagation
from data_loader import load_model


def evaluate_model(parameters, features, labels):
    m = features.shape[0]
    y_hat, _, _ = forward_propagation(features, labels, 0, parameters, ['leaky_relu', 'softmax'])
    predictions = np.argmax(y_hat, axis=0)
    accuracy = np.mean(predictions == labels) * 100
    return accuracy

print("running this file")

if __name__ == "__main__":
    parameters = load_model('models/model1.h5')
    train_features, train_labels, dev_features, dev_labels, test_features, test_labels = initialize_data()
    train_accuracy = evaluate_model(parameters, train_features, train_labels)
    dev_accuracy = evaluate_model(parameters, dev_features, dev_labels)
    test_accuracy = evaluate_model(parameters, test_features, test_labels)

    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Dev accuracy: {dev_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")