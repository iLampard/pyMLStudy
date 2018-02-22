# -*- encoding:utf-8 -*-

import numpy as np
from collections import Counter


def calc_entropy(data_set):
    """
    :param data_set: np.array, 最后一列yes标签
    :return: 香农熵 
    """

    num_data = len(data_set)
    label_dict = Counter(data_set[:, -1])
    entropy = 0.0

    for key in label_dict.keys():
        proba = float(label_dict[key]) / num_data
        entropy += - proba * np.log2(proba)
    return entropy


def split_data_set(data_set, feature_idx, feature_value):
    """
    :param data_set: np.array, 最后一列yes标签
    :param feature_idx: 第几个特征列
    :param feature_value: 特征的值
    :return: 符合该特征的所有实例数据
    """

    ret = []
    for row in data_set:
        if row[feature_idx] == feature_value:
            # 把该列特征删除
            reduced_row = np.hstack((row[:feature_idx], row[feature_idx + 1:]))
            ret.append(reduced_row)

    return np.array(ret)


def calc_cond_entropy(data_set, feature_idx, feature_values):
    """
    :param data_set: np.array, 最后一列yes标签
    :param feature_idx: 第几个特征列
    :param feature_values: 该特征所有可能的取值
    :return: 某特征对应的条件熵
    """

    entropy = 0.0
    for value in feature_values:
        sub_data_set = split_data_set(data_set, feature_idx, value)
        proba = float(len(sub_data_set)) / len(data_set)
        entropy += proba * calc_entropy(sub_data_set)

    return entropy


def calc_information_gain(base_entropy, data_set, feature_idx):
    feature_list = [row[feature_idx] for row in data_set]
    feature_values = set(feature_list)
    return base_entropy - calc_cond_entropy(data_set, feature_idx, feature_values)


def calc_information_gain_rate(base_entropy, data_set, feature_idx):
    return calc_information_gain(base_entropy, data_set, feature_idx) / base_entropy


def create_tree(data_set, method, feature_names):
    labels = data_set[:, - 1]
    # 当类别完全相同时，停止划分，返回该类别
    if len(set(labels)) == 1:
        return labels[0]

    # 当只有一个特征时，返回实例中出现次数最多的类别
    if len(features[0]) == 1:
        return Counter(labels).most_common(1)

    best_feature = choose_feature(data_set, method)
    best_feature_name = feature_names[best_feature]
    best_feature_values = set(data_set[:, best_feature])
    tree = {best_feature_name: {}}
    del feature_names[best_feature]
    for value in best_feature_values:
        sub_feature_names = feature_names[:]
        tree[best_feature_name][value] = create_tree(split_data_set(data_set,
                                                                    best_feature,
                                                                    value),
                                                     method,
                                                     sub_feature_names)

    return tree


def choose_feature(data_set, method):
    num_features = data_set.shape[1] - 1
    base_entropy = calc_entropy(data_set)
    if method.lower() == 'id3':
        information_gain = [calc_information_gain(base_entropy,
                                                  data_set,
                                                  i) for i in range(num_features)]
        return np.argsort(information_gain)[-1]
    elif method.lower() == 'c4.5':
        information_gain_rate = [calc_information_gain_rate(base_entropy,
                                                            data_set,
                                                            i) for i in range(num_features)]
        return np.argsort(information_gain_rate)[-1]
    else:
        raise NotImplementedError


class DecisionTree(object):
    def __init__(self, method='id3'):
        self.tree = None
        self.method = method

    def fit(self, x, y, feature_names):
        data_set = np.hstack((x, y))
        self.tree = create_tree(data_set, method=self.method, feature_names=feature_names)
        return

    def predict(self, x):
        return


if __name__ == '__main__':
    features = [['young', 'no', 'no', 'normal'],
                ['young', 'no', 'no', 'good'],
                ['young', 'yes', 'no', 'good'],
                ['young', 'yes', 'yes', 'normal'],
                ['young', 'no', 'no', 'normal'],
                ['mid', 'no', 'no', 'normal'],
                ['mid', 'no', 'no', 'good'],
                ['mid', 'yes', 'yes', 'good'],
                ['mid', 'no', 'yes', 'verygood'],
                ['mid', 'no', 'yes', 'verygood'],
                ['old', 'no', 'yes', 'verygood'],
                ['old', 'no', 'yes', 'good'],
                ['old', 'yes', 'no', 'good'],
                ['old', 'yes', 'no', 'verygood'],
                ['old', 'no', 'no', 'normal']
                ]
    names = ['years', 'has_job', 'has_house', 'credit']

    labels = [['refuse'],
              ['refuse'],
              ['agree'],
              ['agree'],
              ['refuse'],
              ['refuse'],
              ['refuse'],
              ['agree'],
              ['agree'],
              ['agree'],
              ['agree'],
              ['agree'],
              ['agree'],
              ['agree'],
              ['refuse'],
              ]

    dt = DecisionTree()
    dt.fit(features, labels, names)
    print dt.tree
