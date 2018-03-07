# -*- encoding:utf-8 -*-

import numpy as np
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt

if __name__ == '__main__':
    np.random.seed(0)
    X, y = make_moons(200, noise=0.2)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y)

    plt.show()
