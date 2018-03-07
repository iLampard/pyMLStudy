# -*- encoding:utf-8 -*-

# ref: https://github.com/pangolulu/neural-network-from-scratch

import numpy as np
from abc import ABCMeta, abstractmethod


class MultiplyGate(object):
    @staticmethod
    def forward(w, x):
        return np.dot(w, x)

    @staticmethod
    def backward(w, x, dz):
        dw = np.dot(x, dz)
        dx = np.dot()
        return dw, dx


class AddGate(object):
    @staticmethod
    def forward(x, b):
        return x + b

    @staticmethod
    def backward(w, x, dz):
        db = np.dot()
        return db, dx

# Activation function
class Sigmoid(object):
    @staticmethod
    def forward(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def backward(x, pd_error):
        output = Sigmoid.forward(x)
        return output * (1.0 - output) * pd_error


class Tanh(object):
    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def backward(x, pd_error):
        output = Tanh.forward(x)
        return (1.0 - np.square(output)) * pd_error


# Output
class Softmax(object):
    @staticmethod
    def predict(x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def loss(x, y):
        probs = Softmax.predict(x)
        correct_probs = 0

        return



class Model(object):
