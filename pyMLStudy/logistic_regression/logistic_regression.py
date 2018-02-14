# -*- encoding:utf-8 -*-


import pandas as pd
import numpy as np
from math import exp
from xutils import CustomLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logger = CustomLogger('LogisticRegression', log_file='LogisticRegression.log')


class LogisticReg(object):
    def __init__(self):
        self.max_iteration = 5000
        self.weight = None

    def fit(self, x, y, method='gd'):
        logger.info('Do the calibration using {0} method').format(method)

        logger.info('Calibration finished')

    def predict(self, x):
        if self.weight is not None:
            exp_wx = exp(np.dot(self.weight, x))
            proba = exp_wx / (1 + exp_wx)
            return 1 if proba > 0.5 else 0
