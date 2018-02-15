# -*- encoding:utf-8 -*-


import pandas as pd
import numpy as np
from math import exp
from xutils import CustomLogger
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from argcheck import expect_types

logger = CustomLogger('LogisticRegression', log_file='LogisticRegression.log')


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class LogisticReg(object):
    """
    gradient = sum ( y_i - proba_i) x_i

    """

    def __init__(self):
        self.max_iteration = 5000
        self.learning_rate = 0.0001
        self.weight = None

    @expect_types(x=np.ndarray, y=np.ndarray)
    def fit(self, x, y, method='gd', eps=10e-2):
        y = y.reshape(-1, 1)
        iteration = 0
        logger.info('Do the calibration using {0} method'.format(method))
        num_data = x.shape[0]
        num_feature = x.shape[1]
        X = np.concatenate((x, np.ones((num_data, 1))), axis=1)
        w = np.zeros(num_feature + 1).reshape(-1, 1)
        error = y - sigmoid(np.mat(X) * w)
        while iteration < self.max_iteration or max(error) > eps:
            iteration += 1
            w += self.learning_rate * np.mat(X).T * error
            error = y - sigmoid(np.dot(X, w))

        self.weight = w
        logger.info('Calibration finished: iteration {0} and error {1}'.format(iteration,
                                                                               error))

    @expect_types(x=np.ndarray)
    def predict(self, x):
        if self.weight is not None:
            exp_wx = exp(np.dot(self.weight, x))
            proba = exp_wx / (1 + exp_wx)
            return 1 if proba > 0.5 else 0


if __name__ == '__main__':
    x, labels = make_moons(500, noise=0.2)
    train_features, test_features, train_labels, test_labels = train_test_split(x,
                                                                                labels,
                                                                                test_size=0.33)

    model = LogisticReg()
    model.fit(train_features, train_labels)
