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
    num_step = 10.0
    for i in range(nb_col):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_step
        # for j in range(-1, int(num_step)+1):
        for j in range(0, int(num_step)):
            for inequal in ['lt', 'gt']:
                thresh = range_min + float(j) * step_size
                pred = stump_classify(data_mat, i, thresh, inequal)
                err = np.mat(np.ones((nb_row, 1)))
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


def adaboost_train(data_mat, labels, num_it=40):
    weak_class = []
    m = data_mat.shape[0]
    D = np.mat(np.ones((m, 1)) / m)
    agg_class = np.mat(np.zeros((m, 1)))
    for i in range(num_it):
        best_stump, error, best_class = build_stump(data_mat, labels, D)
        print 'D:', D.T
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-6)))
        best_stump['alpha'] = alpha
        weak_class.append(best_stump)
        print 'class est', best_class.T

        expon = -1.0 * alpha * np.multiply(labels, best_class)
        D = np.multiply(D, np.exp(expon))
        D = D / np.sum(D)

        agg_class += alpha * best_class
        print 'agg_class', agg_class.T
        agg_error = np.multiply(np.sign(agg_class) != labels, np.ones((m, 1)))
        err_rate = np.sum(agg_error) / m
        print '=' * 10
        print 'total error', err_rate
        print
        if err_rate == 0.0:
            break

    return weak_class


if __name__ == "__main__":
    data = np.matrix([[0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                      [1, 3, 2, 1, 2, 1, 1, 1, 3, 2],
                      [3, 1, 2, 3, 3, 2, 2, 1, 1, 1]]).T
    labels = np.array([-1, -1, -1, -1, -1, -1, 1, 1, -1, -1]).reshape((-1, 1))
    D = np.mat(np.ones((10, 1)) / 10)
    print adaboost_train(data, labels, 10)
    # datMat = np.matrix([[1.0, 2.1],
    #                  [2., 1.1],
    #                  [1.3, 1.],
    #                  [1., 1.],
    #                  [2., 1.]])
    #
    # classLabels = np.array([1.0, 1.0, -1.0, -1.0, 1.0]).reshape((-1, 1))
    # # D = np.mat(np.ones((5, 1)) / 5)
    # # print build_stump(data, lables, D)
    # # print build_stump(datMat, classLabels, D)
    # # print adaboost_train(data, labels, 10)
    # print adaboost_train(datMat, classLabels, 10)
    # data = np.matrix([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).T
    # labels = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1]).reshape((-1, 1))
    # print adaboost_train(data, labels, 5)
    #
