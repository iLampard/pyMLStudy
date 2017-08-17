# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import time
import csv
from pyMLStudy.lihongyi_ML2016.HW1.linreg import (gradient_descent_runner,
                                                  predict)


def concat_mat(data, nb_feature=18):
    data_by_date = np.vsplit(data, np.arange(0, len(data), nb_feature))
    data_concat = np.hstack(data_by_date[1:])
    return data_concat


def get_feature_label_mat(data_concat, moving_size, label_row=9):
    mat_list = [data_concat[:, col_start:col_start + moving_size]
                for col_start in range(0, data_concat.shape[1] - moving_size)]
    # change every mat into 1-d array
    feature = [feature.reshape(-1, 1) for feature in mat_list]
    feature = np.hstack(feature)
    label = data_concat[label_row, moving_size:]
    return feature, label


def create_feature_mat_train(csv_file='train.csv', hour_size=9):
    data = pd.read_csv(csv_file, header=0)
    del data['Feature']
    data.replace(to_replace='NR', value=0.0, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    feature = []
    label = []
    date_index = []
    for name, group in data.groupby(pd.TimeGrouper('M')):
        concat_mat_tmp = concat_mat(group.values)
        feature_tmp, label_tmp = get_feature_label_mat(concat_mat_tmp, moving_size=hour_size)
        feature.append(feature_tmp)
        label.append(label_tmp)
        date_index.extend([name] * len(label_tmp))
    feature = np.hstack(feature).astype('float')
    feature = feature.T
    label = np.hstack(label).reshape(-1, 1).astype('float')
    feature_df = pd.DataFrame(feature, index=date_index)
    label_df = pd.DataFrame(label, index=date_index)
    feature_df.to_csv('feature_df.csv')
    label_df.to_csv('label_df.csv')
    return np.mat(feature), np.mat(label)


def create_feature_mat_test(csv_file='test_X.csv', nb_feature=18):
    data = pd.read_csv(csv_file, header=None)
    data.replace(to_replace='NR', value=0.0, inplace=True)
    del data[data.columns[0]]
    del data[data.columns[0]]
    data_split = np.vsplit(data, np.arange(0, len(data), nb_feature))
    data_reshape = [np.ravel(data) for data in data_split[1:]]
    feature = np.vstack(data_reshape).astype('float')
    np.savetxt('feature_test.csv', feature, delimiter=',')
    return feature


if __name__ == "__main__":
    # read train data => create feature matrix and label vector
    feature_train, label_train = create_feature_mat_train()
    # read test data => create feature matrix
    feature_test = create_feature_mat_test()

    # train models
    print 'Training.......'
    t_start = time.time()
    weight = gradient_descent_runner(feature_train, label_train, method='sgd', num_iter=200000)
    print 'Done!'
    print 'Training cost %.3f seconds!' % (time.time() - t_start)

    # make prediction
    pred = predict(feature_test, weight)
    pred = np.array(pred).reshape(-1, ).tolist()
    outputfile = open('kaggle_best.csv', 'wb')
    csv_output = csv.writer(outputfile)

    csv_output.writerow(['id', 'value'])

    for idx, value in enumerate(pred):
        csv_output.writerow(['id_' + str(idx), value])

    outputfile.close()
