#-*- coding:utf-8 -*-

#ref: https://github.com/laisun/mlstudy/blob/master/sklearn/sk_data_reprecessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pyMLStudy.utils import bucketEncode
from pyMLStudy.enum import EncodeType


class DataProcessor(object):
    def __init__(self,
                 labelCol,
                 continuousCols,
                 categoricalCols=None,
                 labelEncode=None,
                 bucketCols=None,
                 bucketBoundaries=None,
                 csvFileDict=None):
        """
        :param labelCols: str/int, label col name
        :param continuousCols: list of str/int - continuous number col names
        :param categoricalCols: list of str/int - categorical col names
        :param labelEncode: enum, indicate the method of mapping of y label str to id. e.g. {'male':0,'female':1 }
        :param bucketCols: list of str/int - non-continuous number col names
        :param bucketBoundaries: list of list - bucket boundaries for each corresponding col
        :param csvFile: dict, path of csv file
        :return:
        """
        self._categoricalCols = categoricalCols
        self._continuousCols = continuousCols
        self._labelCol = labelCol
        self._labelEncode = labelEncode
        self._bucketCols = bucketCols
        self._bucketBoundaries = bucketBoundaries
        self._csvFileDict = csvFileDict
        self._label, self._feature = self.readData()

    def readData(self, csvFileDict=None):
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
    def labelAndFeature(self):
        return self._label, self._feature

