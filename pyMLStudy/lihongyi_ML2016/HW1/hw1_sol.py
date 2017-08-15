# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


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


def create_feature_mat(csv_file='train.csv', hour_size=9):
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


def gradient(X, Y, w):
    dw = X.T * X * w - X.T * Y
    return dw


def adagrad(X, Y, w_0, eta_init=0.1, max_int=500):
    w = w_0
    eta = eta_init
    sum_squre_g = 0
    for i in range(max_int):
        g = gradient(X, Y, w)
        sum_squre_g += g ** 2
        sigma = np.sqrt(sum_squre_g)
        w = w - eta / sigma * g
    return w


def predict(X, w):
    return


def main():
    feature, label = create_feature_mat()
    w = adagrad(X=feature, Y=label, w_0=np.array([0] * feature.shape[1]))
    print w
    return


if __name__ == "__main__":
    main()
