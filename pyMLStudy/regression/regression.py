# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pyMLStudy.dataProcessor import DataProcessor
from pyMLStudy.enum import RegType


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


class Regressor(object):
    def __init__(self, y, x, regType = RegType.LinearReg, trainSize=0.8):
        self._y = y
        self._x = x
        self._regType = regType
        self._trainSize = trainSize
        self._model = None

    def _plot(self, x_test, y_test, y_hat, y_label=['Real data', 'Predict data'], title='Sales prediction by applying regression method'):
        t = np.arange(len(x_test))
        plt.plot(t, y_test, 'r-', linewidth=2, label=y_label[0])
        plt.plot(t, y_hat, 'g-', linewidth=2, label=y_label[1])
        plt.legend(loc='upper right')
        plt.title(title, fontsize=18)
        plt.grid()
        plt.show()


    def regression(self):
        x_train, x_test, y_train, y_test = train_test_split(self._x, self._y, train_size=self._trainSize, random_state=1)
        if self._regType == RegType.LinearReg:
            self._model = Pipeline([('sc', StandardScaler()), ('reg', LinearRegression())])
            self._model.fit(x_train, y_train)
            print 'Calib params：\n', self._model.named_steps['reg'].coef_
            print 'Calib intercept:\n', self._model.named_steps['reg'].intercept_
        else:
            reg = Lasso() if self._regType == RegType.Lasso else Ridge()
            pipeline = Pipeline([('sc', StandardScaler()), ('reg', reg)])
            alpha_can = np.logspace(-3, 2, 10)
            self._model = GridSearchCV(pipeline, param_grid={'alpha': alpha_can}, cv=5)
            self._model.fit(x_train, y_train)
            print 'Hyper param：\n', self._model.best_params_

        y_hat = self._model.predict(x_test)

        mse = (y_hat - y_test) ** 2
        mse = mse.mean()    # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        print mse, rmse

        self._plot(x_test, y_test, y_hat)

        return



def main():
    data = loadData()
    plotData()
    y, x = data.labelAndFeature

    regressor = Regressor(y=y, x=x, regType=RegType.LinearReg)
    regressor.regression()

    return

if __name__ == "__main__":
    main()