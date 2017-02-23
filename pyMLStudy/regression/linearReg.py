# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pyMLStudy.dataProcessor import DataProcessor


mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

_pathDataAdvertising = '..//data//Advertising.csv'


def loadData():
    data = DataProcessor(labelCol='Sales',
                         continuousCols=['TV', 'Radio', 'Newspaper'],
                         csvFile=_pathDataAdvertising)

    return data


def plotData():
    data = pd.read_csv(_pathDataAdvertising)
    plt.figure(figsize=(9,12))
    plt.subplot(311)
    plt.plot(data['TV'], data['Sales'], 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], data['Sales'], 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], data['Sales'], 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    data = loadData()
    plotData()
    y, x = data.labelAndFeature
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    # print x_train, y_train
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print model
    print linreg.coef_
    print linreg.intercept_

    y_hat = linreg.predict(x_test)
    mse = (y_hat - y_test) ** 2
    mse = mse.mean()    # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print mse, rmse

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.grid()
    plt.show()


    return

if __name__ == "__main__":
    main()