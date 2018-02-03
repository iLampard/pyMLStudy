# -*- coding:utf-8 -*-

from pyMLStudy.data_processor import DataProcessor
from pyMLStudy.enum import EncodeType


def load_advertising_data(path):
    data = DataProcessor(label_col='Sales',
                         continuous_cols=['TV', 'Radio', 'Newspaper'],
                         csv_dict={'path': path, 'header': 'infer', 'delim_whitespace': False})

    return data


def load_iris_data(path, onlyUseTwoFeat=False):
    if onlyUseTwoFeat:
        continuous_cols = [0, 1]
    else:
        continuous_cols = [0, 1, 2, 3]

    data = DataProcessor(label_col=4,
                         continuous_cols=continuous_cols,
                         label_encode=EncodeType.LabelEncode,
                         csv_dict={'path': path, 'header': None, 'delim_whitespace': False})
    return data


def load_boston_data(path):
    continuous_cols = range(13)
    data = DataProcessor(label_col=13,
                         continuous_cols=continuous_cols,
                         csv_dict={'path': path, 'header': None, 'delim_whitespace': True})
    return data


def load_bipartition(path):
    continuous_cols = [0, 1]
    data = DataProcessor(label_col=2,
                         continuous_cols=continuous_cols,
                         csv_dict={'path': path, 'header': None, 'delim_whitespace': True})

    return data
