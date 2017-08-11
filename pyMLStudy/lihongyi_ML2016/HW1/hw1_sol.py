# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


def readCSV(data_size):
    data = np.delete(np.transpose(np.genfromtxt("train.csv", delimiter=",")), 0, 1)
    output = []
    out_fin = []
    for i in xrange(0, 3):
        data = np.delete(data, 0, 0)  # delete non-related infos
    for i in xrange(0, 24):
        output.append(np.split(data[i], len(data[i]) / data_size))
    for i in xrange(0, len(output[0])):
        for j in xrange(0, 24):
            out_fin.append(output[j][i])
    out_fin = np.array(out_fin)
    out_fin[np.isnan(out_fin)] = 0
    ##   print "debug nan"
    ##   print len(out_fin)
    print out_fin.shape
    return out_fin


def roll_feature_mat(data_concat, moving_size):
    ret = [data_concat[:, col_start:col_start + moving_size]
           for col_start in range(0, data_concat.shape[1] - moving_size + 1)]
    return ret


def create_feature_mat(hour_size=9):
    data = np.delete(np.genfromtxt('train.csv', delimiter=','), [0, 1], axis=1)
    data = np.delete(data, 0, axis=0)
    data_by_date = np.vsplit(data, np.arange(0, len(data), 18))
    data_concat = np.hstack(data_by_date[1:])
    feature_list = roll_feature_mat(data_concat, moving_size=hour_size)
    # change every mat into 1-d array
    feature = [feature.reshape(-1, 1) for feature in feature_list]
    feature = np.hstack(feature)
    # np.savetxt('feature.csv', feature, delimiter=',')
    print feature[0, :]
    return feature


if __name__ == "__main__":
    create_feature_mat()
