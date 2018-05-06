# -*- encoding:utf-8 -*-

import numpy as np
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from pyMLStudy.models.neutral_network.multi_layer_network import Model


# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y)


if __name__ == '__main__':
    np.random.seed(0)
    X, y = make_moons(200, noise=0.2)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y)

    layer_dim = [2, 3, 2]
    model = Model(layer_dim)
    model.train(X, y, inspect=True)

    # Plot the decision boundary
    plot_decision_boundary(lambda x: model.predict(x), X, y)
    plt.title('Decision Boundary for hidden layer size')

    plt.show()
