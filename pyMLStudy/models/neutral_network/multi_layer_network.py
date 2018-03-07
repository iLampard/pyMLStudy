# -*- encoding:utf-8 -*-

# ref: https://github.com/pangolulu/neural-network-from-scratch

import numpy as np


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
        """
        损失函数： - \sum ln p_i； pi = e^z_i / \sum e^z_i
        """
        num_sample = x.shape[0]
        probs = Softmax.predict(x)
        log_probs = - np.log(probs[range(num_sample), y])
        sum_log_probs = np.sum(log_probs)
        return 1.0 / num_sample * sum_log_probs

    @staticmethod
    def diff(x, y):
        """
        损失函数对 zi 的偏导数
        """
        num_sample = x.shape[0]
        probs = Softmax.predict(x)
        probs[range(num_sample), y] -= 1
        return probs


class Model(object):
    def __init__(self, layer_dim):
        self.w = []
        self.b = []
        for i in range(len(layer_dim) - 1):
            self.w.append(np.random.randn(layer_dim[i], layer_dim[i + 1]))
            self.b.append(np.random.randn(layer_dim[i + 1]).reshape(1, layer_dim[i+1]))


