# -*- encoding:utf-8 -*-

# ref: https://github.com/pangolulu/neural-network-from-scratch

import numpy as np


class MultiplyGate(object):
    @staticmethod
    def forward(w, x):
        """
        :param w: col vector   mul_dim * 1
        :param x: matrix num_sample * mul_dim
        :return:
        """
        return np.dot(x, w)

    @staticmethod
    def backward(w, x, dz):
        """
        :param w: col vector   mul_dim * 1
        :param x: matrix num_sample * mul_dim
        :param dz: col vector num_sample * 1
        :return: dw: col vector mul_dim * 1, dx: matrix num_sample * mul_dim
        """
        dw = np.dot(np.transpose(x), dz)
        dx = np.dot(dz, np.transpose(w))
        return dw, dx


class AddGate(object):
    @staticmethod
    def forward(x, b):
        """
        :param x: col vector   num_sample * 1
        :param b: col vector   num_sample * 1
        :return:
        """
        return x + b

    @staticmethod
    def backward(x, b, dz):
        """
        :param x: matrix num_sample * add_dim
        :param b: col vector   num_sample * 1
        :param dz: col vector num_sample * 1
        """
        dx = dz * np.ones_like(x)
        db = np.dot(np.ones((1, dz.shape[0])), dz)
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
            self.b.append(np.random.randn(layer_dim[i + 1]).reshape(1, layer_dim[i + 1]))

    def calculate_loss(self, x, y):
        mul_gate = MultiplyGate()
        add_gate = AddGate()
        layer = Tanh()
        softmax_output = Softmax()

        input = x
        for i in range(len(self.w)):
            mul = mul_gate.forward(self.w[i], input)
            add = add_gate.forward(mul, self.b[i])
            input = layer.forward(add)

        return softmax_output.loss(input, y)

    def predict(self, x):
        mul_gate = MultiplyGate()
        add_gate = AddGate()
        layer = Tanh()
        softmax_output = Softmax()

        input = x
        for i in range(len(self.w)):
            mul = mul_gate.forward(self.w[i], input)
            add = add_gate.forward(mul, self.b[i])
            input = layer.forward(add)

        probs = softmax_output.predict(input)
        return np.argmax(probs, axis=1)

    def train(self, x, y, num_iteration=20000, eps=0.01, reg_lambda=0.01, inspect=False):
        mul_gate = MultiplyGate()
        add_gate = AddGate()
        layer = Tanh()
        softmax_output = Softmax()

        for index_run in range(num_iteration):
            # forward propagation
            input = x
            forward_operator = [(None, None, input)]

            for i in range(len(self.w)):
                mul = mul_gate.forward(self.w[i], input)
                add = add_gate.forward(mul, self.b[i])
                input = layer.forward(add)
                forward_operator.append((mul, add, input))

            # backward propagation
            dtanh = softmax_output.diff(forward_operator[-1][2], y)

            for i in range(len(forward_operator) - 1, 0, -1):
                dadd = layer.backward(forward_operator[i][1], dtanh)
                db, dmul = add_gate.backward(forward_operator[i][0], self.b[i - 1], dadd)
                dw, dtanh = mul_gate.backward(self.w[i - 1], forward_operator[i - 1][2], dmul)

                dw += reg_lambda * self.w[i - 1]

                self.b[i - 1] -= eps * db
                self.w[i - 1] -= eps * dw

            if inspect and index_run % 1000 == 0:
                print("Loss after iteration {0}: {1}".format(index_run, self.calculate_loss(x, y)))

        return
