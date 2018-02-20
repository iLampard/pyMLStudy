# -*- encoding:utf-8 -*-

import numpy as np
from collections import Counter


def calc_entropy(labels):
    """
    :param labels: np.array
    :return: 香农熵 
    """

    num_data = len(labels)
    label_dict = Counter(labels)
    entropy = 0.0

    for key in label_dict.keys():
        proba = float(label_dict[key]) / num_data
        entropy -= - proba * np.log(proba, 2)
    return entropy


def calc_cond_entropy(labels):

    return


class DecisionTree(object):
    def __init__(self, method='id3'):
        self.tree = None
        self.method = method

    def fit(self, x):
        pass

    def predict(self, x):
        pass
