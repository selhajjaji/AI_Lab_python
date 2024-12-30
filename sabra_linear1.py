# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:56:51 2024

@author: m
"""

import numpy as np
import matplotlib.pyplot as plt

# a.
x = np.random.uniform(-5, 5, 100)

# b.
np.random.seed(28)
# c. Generate y data using the relationship y = 12x - 4
y = 12 * x - 4

# d. Scatter plot of x and y (without noise)
plt.scatter(x, y, alpha=0.5, color='blue')
plt.title("Scatter Plot of y = 12x - 4 (Without Noise)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# e. Inject noise
noise = np.random.normal(loc=0, scale=10, size=100)  
y_noisy = y + noise 

# f. Scatter plot of x and y with noise
plt.scatter(x, y_noisy, alpha=0.5, color='red')
plt.title("Scatter Plot of y = 12x - 4 (With Noise)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()