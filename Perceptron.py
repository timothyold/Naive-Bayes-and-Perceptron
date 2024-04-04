import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
dataset = pd.read_csv('lab02_dataset_1.csv')
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, -1].map({'Positive': 1, 'Negative': -1}).values

# Add a column of ones to X for the bias term
X = np.insert(X, 0, 1, axis=1)

def my_perceptron(X, y, learning_rate=1.0, epochs=1000):
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        error_count = 0
        for xi, target in zip(X, y):
            update = learning_rate * (target - np.where(np.dot(weights, xi) >= 0, 1, -1))
            weights += update * xi
            if update != 0:
                error_count += 1
        if error_count / len(X) < 0.01:  # Misclassification rate less than 1%
            break
    return weights

weights = my_perceptron(X, y)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 1], X[:, 2], X[:, 3], c=y, cmap='bwr')

# Create a mesh to plot the decision surface
xx, yy = np.meshgrid(range(-3, 3), range(-3, 3))
zz = (-weights[1] * xx - weights[2] * yy - weights[0]) / weights[3]
ax.plot_surface(xx, yy, zz, alpha=0.2)

plt.show()
