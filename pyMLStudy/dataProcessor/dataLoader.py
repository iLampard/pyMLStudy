#-*- coding:utf-8 -*-

from pyMLStudy.dataProcessor import DataProcessor
from pyMLStudy.enum import EncodeType



def loadAdvertisingData(path):
    data = DataProcessor(labelCol='Sales',
                         continuousCols=['TV', 'Radio', 'Newspaper'],
                         csvFileDict={'path': path, 'header': 'infer', 'delim_whitespace': False})

    return data


def loadIrisData(path, onlyUseTwoFeat=False):
    if onlyUseTwoFeat:
        continuousCols = [0, 1]
    else:
        continuousCols = [0, 1, 2, 3]

    data = DataProcessor(labelCol=4,
                         continuousCols=continuousCols,
                         labelEncode=EncodeType.LabelEncode,
                         csvFileDict={'path': path, 'header': None, 'delim_whitespace': False})
    return data


def loadBostonData(path):
    continousCols = range(13)
    data = DataProcessor(labelCol=13,
                         continuousCols=continousCols,
                         csvFileDict={'path': path, 'header': None, 'delim_whitespace': True})
    return data