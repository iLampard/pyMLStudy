# -*- coding:utf-8 -*-

from enum import IntEnum
from enum import unique


@unique
class EncodeType(IntEnum):
    LabelEncode = 1
    OneHotEncode = 2