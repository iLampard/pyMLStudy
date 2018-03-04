# -*- encoding:utf-8 -*-

# ref: https://github.com/pangolulu/neural-network-from-scratch

import numpy as np
from abc import ABCMeta, abstractmethod


class Gate(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, w, x):
        pass

    @abstractmethod
    def backward(self, w, x, gradient_next):
        pass


class MultiplyGate(Gate):
    def forward(self, w, x):
        return np.dot(w, x)

    def backward(self, w, x, gradient_next):
        pass


class AddGate(Gate):
    def forward(self, x, b):
        return x + b

    def backward(self, w, x, gradient_next):
        db = np.dot()
        return db, dx
