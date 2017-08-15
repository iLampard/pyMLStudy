# -*- coding:utf-8 -*-


import numpy as np

EPS = 1e-8


class LinReg(object):
    """
    https://github.com/mob5566/ML2016/blob/master/hw1/linreg_model.py
    """

    def __init__(self, max_iter=100, eta=1e-2, method='sgd', scale=False, batch_size=100, intercept=False):
        self.max_iter = max_iter
        self.eta = eta
        self.method = method
        self.scale = scale
        self.batch_size = batch_size
        self.intercept = intercept
        self.params = None
        self.acc_dw = None

    def fit(self, X, y):
        # check whether the training data is empty or not
        if len(X) <= 0 or len(y) <= 0:
            raise AssertionError

        # check whether the size of training data (X, y) is consistent or not
        if len(X) != len(y):
            raise AssertionError

        X = np.array(X)
        y = np.array(y)

        if self.intercept:
            self.params = np.random.rand(X.shape[1] + 1)
            self.acc_dw = np.zeros(X.shape[1] + 1)
        else:
            self.params = np.random.rand(X.shape[1])
            self.acc_dw = np.zeros(X.shape[1])

        if self.scale:
            X = (X - np.mean(X, axis=0) / np.std(X, axis=0))

        for i in range(self.max_iter):
            if self.method == 'sgd':
                pass
            elif self.method == 'adagrad':
                self._gradient_descent(X, y)

    def predict(self, X):

        return

    def _gradient_descent(self, X, y):
        dw = gradient(X, y, intercept=self.intercept)
        if self.method == 'adagrad':
            self.acc_dw += dw ** 2
            dw /= np.sqrt(self.acc_dw + EPS)
        self.params = self.params - dw * self.eta


def RMSE(Y, Y_hat):
    return np.sqrt(np.sum(Y_hat - Y) ** 2 / len(Y))


def gradient(X, y, w, intercept=False):
    if intercept:
        X = np.hstack((X, np.array([1] * len(X)).reshape(-1, 1)))
    dw = X.T * X * w - X.T * y
    return dw
