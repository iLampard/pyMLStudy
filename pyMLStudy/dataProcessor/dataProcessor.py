#-*- coding:utf-8 -*-

#ref: https://github.com/laisun/mlstudy/blob/master/sklearn/sk_data_reprecessing.py

import pandas as pd
from pyMLStudy.utils import bucketEncode


class DataProcessor(object):
    def __init__(self,
                 labelCol,
                 continuousCols,
                 categoricalCols=None,
                 label2Number=None,
                 bucketCols=None,
                 bucketBoundaries=None,
                 csvFile=None):
        """
        :param labelCols: str, label col name
        :param continuousCols: list of str - continuous number col names
        :param categoricalCols: list of str - categorical col names
        :param label2Number: dict, the mapping of y label str to id. e.g. {'male':0,'female':1 }
        :param bucketCols: list of str - non-continuous number col names
        :param bucketBoundaries: list of list - bucket boundaries for each corresponding col
        :param csvFile: str, path of csv file
        :return:
        """
        self._categoricalCols = categoricalCols
        self._continuousCols = continuousCols
        self._labelCol = labelCol
        self._label2Number = label2Number
        self._bucketCols = bucketCols
        self._bucketBoundaries = bucketBoundaries
        self._csvFile = csvFile
        self._label, self._feature = self.readData()

    def readData(self, csvFile=None):
        """
        :return: pd.DataFrame, label and fature
        """
        csvFile = self._csvFile if csvFile is None else csvFile
        data = pd.read_csv(csvFile)

        y = data[self._labelCol]
        if self._label2Number is not None:
            y = y.apply(lambda x: self._label2Number[x])

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

        feature = pd.concat([ xContinuousNum, xCate, xBucket], axis=1)
        return y, feature

    @property
    def labelAndFeature(self):
        return self._label, self._feature

