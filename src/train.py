import numpy as np
import math
import h5py
import struct
import matplotlib.pyplot as plt
from data_loader import *
from activations import *
from initialize import *
from model import *
from evaluate import evaluate_model
from time import sleep


# Initialize the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
costs = []
x_vals = []

# Set up the plot
line, = ax.plot([], [], 'r-')  # 'r-' stands for a red line
ax.set_xlim(0, 10)  # Update based on the number of iterations
ax.set_ylim(0, 5)   # Update based on expected cost range

layer_dims = [4, 4, 3]
location = 'models/model1.h5'

# If you want to load a previously saved model
try:
    parameters = load_model(location)
except:
    # First time training: Initialize parameters if no model is loaded
    parameters = initialize_parameters(layer_dims, ['he', 'xavier'])

mini_batches, train_features, train_labels, dev_features, dev_labels, test_features, test_labels = initialize_data(3, 32)

# Set the total number of iterations (adjust as necessary)
total_iterations = 100
iteration_counter = 0
t=0
adam_velocity, adam_rms = initialize_adam_velocity(parameters)
lr = 0.005
lambd = 0.003
decay_rate = 0.001

for epoch in range(400):  # Number of epochs
    parameters, batch_costs, total_cost, t, adam_velocity, adam_rms = nn_model(mini_batches, lr, lambd, decay_rate, parameters, adam_velocity, adam_rms, t=t,
                                                                  layer_dims=layer_dims, activications=['leaky_relu', 'softmax'], print_cost=False)
    
    # Store costs and update the plot
    costs.extend(batch_costs)
    x_vals.extend(list(range(iteration_counter, iteration_counter + len(batch_costs))))  # Correctly track each mini-batch iteration
    iteration_counter += len(batch_costs)  # Update iteration counter
    total_cost = total_cost / len(mini_batches)/64
    print(total_cost)
    line.set_data(x_vals, costs)  # Update line data

    ax.set_xlim(0, iteration_counter)  # Dynamically adjust x-axis based on number of iterations
    ax.set_ylim(0, max(costs) + 0.1)  # Adjust y-axis based on the maximum cost
    
    plt.draw()  # Redraw the plot
    plt.pause(0.1)  # Pause to allow real-time updates
    
    # Save model every 10 iterations
    if (epoch + 1) % 10 == 0:
        save_model(parameters, location)

train_accuracy = evaluate_model(parameters, train_features, train_labels)
dev_accuracy = evaluate_model(parameters, dev_features, dev_labels)

# Present the final model accuracy
print(f"Learning rate: {lr}")
print(f"Lambda: {lambd}")
print(f"Decay rate: {decay_rate}")
print(f"Train accuracy: {train_accuracy:.2f}%")
print(f"Dev accuracy: {dev_accuracy:.2f}%")

plt.ioff()  # Turn off interactive mode
plt.show() 