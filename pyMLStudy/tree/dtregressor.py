# -*- coding:utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor



def linearReg():
    N = 100
    x = np.random.rand(N) * 6 - 3     # [-3,3)
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    print y.shape
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的
    print x.shape

    reg = DecisionTreeRegressor(criterion='mse', max_depth=9)
    dt = reg.fit(x, y)
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    y_hat = dt.predict(x_test)
    plt.plot(x, y, 'r*', ms=10, label='Actual')
    plt.plot(x_test, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # 比较决策树的深度影响
    depth = [2, 4, 6, 8, 10]
    clr = 'rgbmy'
    reg = [DecisionTreeRegressor(criterion='mse', max_depth=depth[0]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[1]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[2]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[3]),
           DecisionTreeRegressor(criterion='mse', max_depth=depth[4])]

    plt.plot(x, y, 'k^', linewidth=2, label='Actual')
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    for i, r in enumerate(reg):
        dt = r.fit(x, y)
        y_hat = dt.predict(x_test)
        plt.plot(x_test, y_hat, '-', color=clr[i], linewidth=2, label='Depth=%d' % depth[i])
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def multiOutputReg():
    N = 400
    x = np.random.rand(N) * 8 - 4     # [-4,4)
    x.sort()
    y1 = np.sin(x) + 3 + np.random.randn(N) * 0.1
    y2 = np.cos(0.3*x) + np.random.randn(N) * 0.01
    # y1 = np.sin(x) + np.random.randn(N) * 0.05
    # y2 = np.cos(x) + np.random.randn(N) * 0.1
    # y1 = 16 * np.sin(x) ** 3 + np.random.randn(N)
    # y2 = 13 * np.cos(x) - 5 * np.cos(2*x) - 2 * np.cos(3*x) - np.cos(4*x) + 0.1* np.random.randn(N)
    print y1
    print y2
    y = np.vstack((y1, y2)).T
    print y
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的

    deep = 5
    reg = DecisionTreeRegressor(criterion='mse', max_depth=deep)
    dt = reg.fit(x, y)

    x_test = np.linspace(-4, 4, num=1000).reshape(-1, 1)
    print x_test
    y_hat = dt.predict(x_test)
    print y_hat
    plt.scatter(y[:, 0], y[:, 1], c='r', s=40, label='Actual')
    plt.scatter(y_hat[:, 0], y_hat[:, 1], c='g', marker='s', s=100, label='Depth=%d' % deep, alpha=1)
    plt.legend(loc='upper left')
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    multiOutputReg()