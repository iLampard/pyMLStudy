# -*- coding:utf-8 -*-

from enum import IntEnum
from enum import unique


@unique
class RegType(IntEnum):
    LinearReg = 1
    Lasso = 2
    Ridge = 3
    LogisticReg = 4