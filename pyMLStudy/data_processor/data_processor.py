# -*- coding:utf-8 -*-

# ref: https://github.com/laisun/mlstudy/blob/master/sklearn/sk_data_reprecessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pyMLStudy.utils import bucketEncode
from pyMLStudy.enum import EncodeType


class DataProcessor(object):
    def __init__(self,
                 label_col,
                 continuous_cols,
                 categorical_cols=None,
                 label_encode=None,
                 bucket_cols=None,
                 bucket_boundaries=None,
                 csv_dict=None):
        """
        :param labelCols: str/int, label col name
        :param continuous_cols: list of str/int - continuous number col names
        :param categorical_cols: list of str/int - categorical col names
        :param label_encode: enum, indicate the method of mapping of y label str to id. e.g. {'male':0,'female':1 }
        :param bucket_cols: list of str/int - non-continuous number col names
        :param bucket_boundaries: list of list - bucket boundaries for each corresponding col
        :param csvFile: dict, path of csv file
        :return:
        """
        self._categoricalCols = categorical_cols
        self._continuousCols = continuous_cols
        self._labelCol = label_col
        self._labelEncode = label_encode
        self._bucketCols = bucket_cols
        self._bucketBoundaries = bucket_boundaries
        self._csvFileDict = csv_dict
        self._label, self._feature = self.read_data()

    def read_data(self, csvFileDict=None):
        """
        :return: pd.DataFrame, label and fature
        """
        csvFileDict = self._csvFileDict if csvFileDict is None else csvFileDict
        data = pd.read_csv(csvFileDict['path'],
                           header=csvFileDict['header'],
                           delim_whitespace=csvFileDict['delim_whitespace'])

        y = data[self._labelCol]
        if self._labelEncode is not None:
            if self._labelEncode == EncodeType.LabelEncode:
                le = LabelEncoder()
                y = le.fit_transform(y)
            else:
                raise NotImplementedError

        xContinuousNum = data[self._continuousCols]

        if self._categoricalCols is not None:
            xCate = data[self._categoricalCols]
        else:
            xCate = None

        if self._bucketCols is not None:
            xBucketTemp = data[self._bucketCols]
            xBucket = pd.DataFrame()
            for i in xBucketTemp.columns:
                colBucket = xBucket[i].apply(bucketEncode, args=(self._bucketBoundaries[i],))
                xBucket = pd.concat([xBucket, colBucket], axis=1)
        else:
            xBucket = None

        feature = pd.concat([xContinuousNum, xCate, xBucket], axis=1)
        return y, feature

    @property
    def label_feature(self):
        return self._label, self._feature
