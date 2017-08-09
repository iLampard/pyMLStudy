# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from PIL import Image


def rank_col(data_file, col):
    data = pd.read_csv(data_file, sep='\s+', header=None)
    data_col = data[data.columns[col]]
    ret = data_col.sort_values().values.tolist()
    return ret


def reverse_img(img_file):
    img = Image.open(img_file)
    new_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    new_img.show()
    new_img.save('new_lena.png')
    return


if __name__ == "__main__":
    print rank_col('hw0_data.dat', 1)
    reverse_img('lena.png')
