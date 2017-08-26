# -*- coding:utf-8 -*-

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def MSE(X, y, weight):
    error = sigmoid(np.mat(X) * weight) - y
    ret = error.T * error / float(len(X))
    return ret


def update_weight(X, y, weight, alpha, method):
    """
    :param alpha: float, learning rate
    :return:
    """
    gradient = calc_gradient(X, y, weight, method)
    weight += gradient * alpha
    return weight


def calc_gradient(X, y, weight, method):
    if method == 'gd':
        gradient = np.mat(X).T * (y - sigmoid(np.mat(X) * weight))
    elif method == 'sgd':
        rand_index = int(np.random.uniform(0, len(X)))
        gradient = np.mat(X[rand_index]).T * (sigmoid(np.mat(X[rand_index]) * weight) - y[rand_index])
    else:
        raise NotImplementedError
    #gradient /= float(len(X))
    return gradient


def gradient_descent_runner(X, y, weight_init=None, alpha=0.001, num_iter=500, method='gd'):
    weight = weight_init if weight_init is not None else np.ones(X.shape[1]).reshape(-1, 1)
    for i in range(num_iter):
        weight = update_weight(X, y, weight, alpha, method)
        mse = MSE(X, y, weight)
        print "Iteration {0} - weight = {1}, error = {2}".format(i, weight, mse)
    return weight


def plot_fit(weight, data):
    import matplotlib.pyplot as plt
    xcord_1 = []
    ycord_1 = []
    xcord_2 = []
    ycord_2 = []
    n = data.shape[0]
    for i in range(n):
        if int(data[i, 2]) == 1:
            xcord_1.append(data[i, 0])
            ycord_1.append(data[i, 1])
        else:
            xcord_2.append(data[i, 0])
            ycord_2.append(data[i, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord_1, ycord_1, s=30, c='red', marker='s')
    ax.scatter(xcord_2, ycord_2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weight[0] - weight[1] * x) / weight[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('Y1')
    plt.show()


def run():
    data = np.genfromtxt("testSet.txt")
    weight_init = np.array([1.0, 1.0, 1.0]).reshape(-1, 1)
    num_iter = 500
    method = 'gd'
    X = np.hstack((np.ones(len(data)).reshape(-1, 1), data[:, 0:2]))
    y = data[:, 2].reshape(-1, 1)

    print "Starting gradient descent at weight={0}, error = {1}, method is {2}". \
       format(weight_init, MSE(X, y, weight_init), method)
    print "Running..."
    weight = gradient_descent_runner(X, y, weight_init, method=method, num_iter=num_iter)
    print "After {0} iterations weight{1}, error = {2}".format(num_iter, weight,
                                                               MSE(X, y, weight))
    plot_fit(weight, data)

if __name__ == '__main__':
    run()



