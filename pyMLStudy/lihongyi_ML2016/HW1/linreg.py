# -*- coding:utf-8 -*-


import numpy as np

EPS = 1e-5


class LinReg(object):
    """
    https://github.com/mob5566/ML2016/blob/master/hw1/linreg_model.py
    """

    def __init__(self, max_iter=100, eta=1e-2, method='sgd', scale=False, batch_size=100):
        self.max_iter = max_iter
        self.eta = eta
        self.method = method
        self.scale = scale
        self.batch_size = batch_size
        self.params = None
        self.acc_dw = None
        self.hist_error = []

    def fit(self, X, y):
        # check whether the training data is empty or not
        if len(X) <= 0 or len(y) <= 0:
            raise AssertionError

        # check whether the size of training data (X, y) is consistent or not
        if len(X) != len(y):
            raise AssertionError

        X = np.array(X)
        y = np.array(y)

        self.params = np.random.rand(X.shape[1]).reshape(-1, 1)
        self.acc_dw = 0.0

        if self.scale:
            X_scale = []
            for column in X.T:
                if np.std(column) > EPS:
                    column = (column - np.mean(column)) / np.std(column)
                X_scale.append(column.reshape(-1, 1))
            X = np.hstack(X_scale)

        for i in range(self.max_iter):
            if self.method == 'sgd':
                rmask = range(len(X))
                np.random.shuffle(rmask)
                X_mask = X[rmask]
                y_mask = y[rmask]
                for i in range(0, len(X) - self.batch_size, self.batch_size):
                    X_tmp = X_mask[i:i + self.batch_size]
                    y_tmp = y_mask[i:i + self.batch_size]
                    self._gradient_descent(X_tmp, y_tmp)
            else:
                self._gradient_descent(X, y)

            self.hist_error.append(RMSE(self.predict(X), y))
            print self.hist_error[-1]

    def predict(self, X):
        return np.dot(X, self.params)

    def _gradient_descent(self, X, y):
        dw = gradient(X, y, self.params)
        if self.method == 'adagrad':
            self.acc_dw += dw ** 2
            dw /= np.sqrt(self.acc_dw + EPS)
        self.params = self.params - dw * self.eta


def RMSE(Y, Y_hat):
    return np.sqrt(np.sum((Y_hat - Y) ** 2) / len(Y))


def gradient(X, y, w):
    X = np.mat(X)
    w = w.reshape(-1, 1)
    y = y.reshape(-1, 1)
    dw = X.T * X * w - X.T * y
    return dw
