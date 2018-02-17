# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


def single_output_example():
    N = 100
    x = np.random.rand(N) * 6 - 3  # [-3,3)
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的

    reg = DecisionTreeRegressor(criterion='mse', max_depth=9)
    reg.fit(x, y)

    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    y_hat = reg.predict(x_test)
    plt.plot(x, y, 'r*', ms=10, label='Actual')
    plt.plot(x_test, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

def multiple_output_example():
    N = 400
    x = np.random.rand(N) * 8 - 4  # [-4,4)
    x.sort()
    y1 = np.sin(x) + 3 + np.random.randn(N) * 0.1
    y2 = np.cos(0.3 * x) + np.random.randn(N) * 0.01
    # y1 = np.sin(x) + np.random.randn(N) * 0.05
    # y2 = np.cos(x) + np.random.randn(N) * 0.1
    # y1 = 16 * np.sin(x) ** 3 + np.random.randn(N)
    # y2 = 13 * np.cos(x) - 5 * np.cos(2*x) - 2 * np.cos(3*x) - np.cos(4*x) + 0.1* np.random.randn(N)
    y = np.vstack((y1, y2)).T
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的

    deep = 5
    reg = DecisionTreeRegressor(criterion='mse', max_depth=deep)
    dt = reg.fit(x, y)

    x_test = np.linspace(-4, 4, num=1000).reshape(-1, 1)
    y_hat = dt.predict(x_test)
    plt.scatter(y[:, 0], y[:, 1], c='r', s=40, label='Actual')
    plt.scatter(y_hat[:, 0], y_hat[:, 1], c='g', marker='s', s=100, label='Depth=%d' % deep, alpha=1)
    plt.legend(loc='upper left')
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    multiple_output_example()
