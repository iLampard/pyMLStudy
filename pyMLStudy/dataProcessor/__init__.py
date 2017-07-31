#-*- coding:utf-8 -*-


from pyMLStudy.dataProcessor.dataProcessor import DataProcessor
from pyMLStudy.dataProcessor.dataLoader import loadBostonData
from pyMLStudy.dataProcessor.dataLoader import loadIrisData
from pyMLStudy.dataProcessor.dataLoader import loadAdvertisingData
from pyMLStudy.dataProcessor.dataLoader import loadBipartition


__all__ = ['DataProcessor',
           'loadBostonData',
           'loadIrisData',
           'loadAdvertisingData',
           'loadBipartition']