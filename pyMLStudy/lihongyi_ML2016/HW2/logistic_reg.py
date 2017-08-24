# -*- coding:utf-8 -*-

import numpy as np


def sigmoid(X, weight):
    z = np.mat(X) * weight
    return 1.0 / (1 + np.exp(-z))


def MSE(X, y, weight):
    error = sigmoid(X, weight) - y
    ret = np.sum(error.T * error) / float(len(X))
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
        gradient = np.mat(X).T * (sigmoid(X, weight) - y)
    elif method == 'sgd':
        rand_index = int(np.random.uniform(0, len(X)))
        gradient = np.mat(X[rand_index]).T * (sigmoid(X[rand_index], weight)-y[rand_index])
    else:
        raise NotImplementedError
    gradient /= float(len(X))
    return gradient


def gradient_descent_runner(X, y, weight_init=None, alpha=0.0001, num_iter=100, method='gd'):
    weight = weight_init if weight_init is not None else np.ones(X.shape[1]).reshape(-1, 1)
    for i in range(num_iter):
        weight = update_weight(X, y, weight, alpha, method)
        mse = MSE(X, y, weight)
        print "Iteration {0} - error = {1}".format(i, mse)
    return weight

