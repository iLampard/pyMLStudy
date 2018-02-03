#-*- coding:utf-8 -*-


from pyMLStudy.data_processor.data_processor import DataProcessor
from pyMLStudy.data_processor.data_loader import load_boston_data
from pyMLStudy.data_processor.data_loader import load_iris_data
from pyMLStudy.data_processor.data_loader import load_advertising_data
from pyMLStudy.data_processor.data_loader import load_bipartition


__all__ = ['DataProcessor',
           'load_boston_data',
           'load_iris_data',
           'load_advertising_data',
           'load_bipartition']