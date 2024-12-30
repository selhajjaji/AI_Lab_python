# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:57:35 2024

@author: m
"""
##################################################################
#Exercise # 1: Single layer feed forward to recognize sum pattern 
##################################################################
import numpy as np
# Set seed for reproducibility
np.random.seed(1)
# Generate two sets of random numbers between -0.6 and +0.6
input_team6 = np.random.uniform(-0.6, 0.6, (10, 2))
print("Input Data:",input_team6)
output_team6 = input_team6.sum(axis=1, keepdims=True)
print("Output Data:")
print(output_team6)
import neurolab as nl
# Define the network :Single-layer feed-forward network
net = nl.net.newff([[input_team6.min(), input_team6.max()]] * 2, [6, 1])
# Set training parameters
net.trainf = nl.train.train_gd  # Gradient descent
# Train the network
error = net.train(input_team6, output_team6, epochs=500, show=15, goal=0.00001)
# Test the network with test values
test_values = np.array([[0.1, 0.2]])
result_1 = net.sim(test_values)
print("Result #1 (Single Layer):", result_1)
actual = 0.1 + 0.2
print("Actual Result:", actual)
################################################################
#Exercise # 2: Multi-layer feed forward to recognize sum pattern 
################################################################
import numpy as np
import neurolab as nl
# Step 1: Generate Training Data
np.random.seed(1)  # Set seed for reproducibility
input_team6 = np.random.uniform(-0.6, 0.6, (10, 2))
output_team6 = input_team6.sum(axis=1, keepdims=True)  # Sum of inputs as output
# Step 2: Create a Multi-Layer Neural Network
# Define input and output ranges
input_minmax = [[input_team6.min(), input_team6.max()]] * 2
# Create the network with two hidden layers (5 and 3 neurons) and one output
net = nl.net.newff(input_minmax, [5, 3, 1])
# Set training algorithm to Gradient Descent Backpropagation
net.trainf = nl.train.train_gd
# Step 3: Train the Network
error = net.train(input_team6, output_team6, epochs=1000, show=100, goal=0.00001)
# Step 4: Test the Network
test_values = np.array([[0.1, 0.2]])
result_2 = net.sim(test_values)
print("Result #2 (Multi Layer):", result_2)
# Comparison and findings
actual = 0.1 + 0.2
print("Actual Result:", actual)
##########################################################################################################
#Exercise # 3: Define the network :Single-layer feed-forward network
##########################################################################################################import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

# Step 1: Generate 100 training instances
np.random.seed(1)
input_team = np.random.uniform(-0.6, 0.6, size=(100, 2))
output_team = input_team.sum(axis=1, keepdims=True)

# Single-layer feed-forward network
net1 = nl.net.newff([[min(input_team[:, 0]), max(input_team[:, 0])],
                     [min(input_team[:, 1]), max(input_team[:, 1])]], [6, 1])
net1.trainf = nl.train.train_gd
error1 = net1.train(input_team, output_team, epochs=500, show=15, goal=0.00001)
result3 = net1.sim([[0.1, 0.2]])
print("Result #3 (Single Layer):", result3)
##########################################################################################################
# Exercise #4: Multi-layer feed-forward network
##########################################################################################################import numpy as np
# Step 1: Generate 100 training instances
np.random.seed(1)
input_team = np.random.uniform(-0.6, 0.6, size=(100, 2))
output_team = input_team.sum(axis=1, keepdims=True)
net2 = nl.net.newff([[min(input_team[:, 0]), max(input_team[:, 0])],
                     [min(input_team[:, 1]), max(input_team[:, 1])]], [5, 3, 1])
net2.trainf = nl.train.train_gd
error2 = net2.train(input_team, output_team, epochs=1000, show=100, goal=0.00001)
result4 = net2.sim([[0.1, 0.2]])
print("Result #4 (Multi Layer):", result4)
# Plot Error vs Epoch
plt.plot(error1, label="Single Layer Error")
plt.plot(error2, label="Multi Layer Error")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error vs Epoch")
plt.legend()
plt.show()
# Compare results
actual = 0.1 + 0.2
print("Actual Result:", actual)
##########################################################################################################
 #Exercise # 5: Three input multi-layer feed forward to recognize sum pattern with more training data 
##########################################################################################################import numpy as np

import numpy as np
import neurolab as nl

# Generate training data
np.random.seed(1)
input_team6 = np.random.uniform(-0.6, 0.6, (100, 3))  # 100 samples, 3 inputs
output_team6 = input_team6.sum(axis=1, keepdims=True)  # Target output: sum of inputs

# Create and train a single-layer network
net1 = nl.net.newff([[input_team6.min(), input_team6.max()]] * 3, [6, 1])
net1.trainf = nl.train.train_gd
error1 = net1.train(input_team6, output_team6, epochs=500, show=100, goal=0.00001)

# Test the network
test_values = np.array([[0.2, 0.1, 0.2]])
result5 = net1.sim(test_values)
print("Result #5 (Single Layer):", result5)
# Create and train a multi-layer network
net2 = nl.net.newff([[input_team6.min(), input_team6.max()]] * 3, [5, 3, 1])
net2.trainf = nl.train.train_gd
error2 = net2.train(input_team6, output_team6, epochs=1000, show=100, goal=0.00001)

# Test the network
result6 = net2.sim(test_values)
print("Result #6 (Multi Layer):", result6)