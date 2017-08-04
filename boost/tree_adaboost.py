# -*- coding:utf-8 -*-

import numpy as np


# ref: 机器学习实战

def stump_classify(data_mat, dim, thresh, direction):
    ret_array = np.ones((data_mat.shape[0], 1))
    if direction == 'lt':
        ret_array[data_mat[:, dim] <= thresh] = -1.0
    else:
        ret_array[data_mat[:, dim] > thresh] = -1.0
    return ret_array


def build_stump(data_mat, labels, D):
    nb_row, nb_col = data_mat.shape
    best_stump = {}
    best_class = np.zeros((nb_row, 1))
    min_err = np.inf
    num_step = 5
    for i in range(nb_col):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_step
        for j in range(-1, int(num_step) + 1):
            for inequal in ['lt', 'gt']:
                thresh = range_min + float(j) * step_size
                pred = stump_classify(data_mat, i, thresh, inequal)
                err = np.ones((nb_row, 1))
                err[pred == labels] = 0
                weight_err = D.T * err
                print "split: dim %d, thresh % .2f, thresh inequal: %s, the weighted error is %.3f" % (
                    i, thresh, inequal, weight_err)
                if weight_err < min_err:
                    min_err = weight_err
                    best_class = pred.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh
                    best_stump['ineq'] = inequal

    return best_stump, min_err, best_class


if __name__ == "__main__":
    data = np.matrix([[0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                      [1, 3, 2, 1, 2, 1, 1, 1, 3, 2],
                      [3, 1, 2, 3, 3, 2, 2, 1, 1, 1]]).T
    lables = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1])

