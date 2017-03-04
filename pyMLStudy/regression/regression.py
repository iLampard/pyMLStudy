# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from pyMLStudy.enum import RegType
from pyMLStudy.dataProcessor import loadAdvertisingData
from pyMLStudy.dataProcessor import loadBostonData
from pyMLStudy.dataProcessor import loadIrisData

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

_pathAdvertisingData = '..//data//Advertising.csv'
_pathHousingData = '..//data//housing.data'
_pathIrisData = '..//data//Iris.data'



def plotAdvertisingData():
    data = pd.read_csv(_pathAdvertisingData)
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
    def __init__(self, y, x, regType = RegType.LinearReg, trainSize=0.8, plotTitle=None):
        self._y = y
        self._x = x
        self._regType = regType
        self._trainSize = trainSize
        self._plotTitle = plotTitle
        self._model = None

    def _plot(self, x_test, y_test, y_hat,
              y_data_label=['Real data', 'Predict data'],
              title='Sales prediction by applying regression method',
              xlabel=None,
              ylabel=None):
        t = np.arange(len(x_test))
        plt.plot(t, y_test, 'r-', linewidth=2, label=y_data_label[0])
        plt.plot(t, y_hat, 'g-', linewidth=2, label=y_data_label[1])
        plt.legend(loc='best')
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=15)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=15)
        plt.title(title, fontsize=18)
        plt.grid()
        plt.show()


    def _plotClassfier(self, x, y, model, xlable=u'花萼长度', ylabel=u'花萼宽度', title=u'鸢尾花Logistic回归分类效果 - 标准化'):
        N, M = 500, 500     # 横纵各采样多少个值
        x1_min, x1_max = x.loc[:, 0].min(), x.loc[:, 0].max()   # 第0列的范围
        x2_min, x2_max = x.loc[:, 1].min(), x.loc[:, 1].max()   # 第1列的范围
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, M)
        x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
        x_test = np.stack((x1.flat, x2.flat), axis=1)   # 测试点

        cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        y_hat = model.predict(x_test)                  # 预测值
        y_hat = y_hat.reshape(x1.shape)                 # 使之与输入的形状相同
        plt.figure(facecolor='w')
        plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)     # 预测值的显示
        plt.scatter(x.loc[:, 0], x.loc[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)    # 样本的显示
        plt.xlabel(xlable, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid()
        plt.title(title, fontsize=17)
        plt.show()

        return


    def regression(self):
        x_train, x_test, y_train, y_test = train_test_split(self._x, self._y, train_size=self._trainSize, random_state=1)

        if self._regType == RegType.LinearReg:
            self._model = Pipeline([('ss', StandardScaler()), ('reg', LinearRegression())])
            self._model.fit(x_train, y_train)
            print 'Calib params：\n', self._model.get_params('reg')['reg'].coef_
            print 'Calib intercept:\n', self._model.named_steps['reg'].intercept_

        elif self._regType == RegType.Lasso or self._regType == RegType.Ridge:
            reg = LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False) if self._regType == RegType.Lasso else \
                RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False)
            self._model = Pipeline([('ss', StandardScaler()), ('reg', reg)])
            self._model.fit(x_train, y_train)
            print 'Hyper param：\n', self._model.get_params('reg')['reg'].best_params_

        elif self._regType == RegType.LogisticReg:
            self._model = Pipeline([('ss', StandardScaler()), ('reg', LogisticRegression()) ])
            self._model.fit(x_train, y_train)

        elif self._regType == RegType.ElasticNet:
            self._model = Pipeline([('ss', StandardScaler()),
                                    ('poly', PolynomialFeatures(degree=3, include_bias=True)),
                                    ('reg', ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1], alphas=np.logspace(-3, 2, 5),
                                                            fit_intercept=False, max_iter=1e3, cv=3))
                                ])
            self._model.fit(x_train, y_train)
            linear = self._model.get_params('reg')['reg']
            print 'Hyper param：\n', linear.alpha_
            print 'L1 ratio：', linear.l1_ratio_


        y_hat = self._model.predict(x_test)

        if self._regType == RegType.LinearReg or self._regType == RegType.Lasso or self._regType == RegType.Ridge:
            mse = mean_squared_error(y_test, y_hat)    # Mean Squared Error
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            print mse, rmse
            self._plot(x_test, y_test, y_hat, self._plotTitle)

        elif self._regType == RegType.LogisticReg:
            y_hat_prob = self._model.predict_proba(x_test)
            np.set_printoptions(suppress=True)
            print 'y_hat = \n', y_hat
            print 'y_hat_prob = \n', y_hat_prob
            print u'准确度：%.2f%%' % (100 * (y_hat == y_test).mean())
            if len(x_test.columns) == 2:
                self._plotClassfier(x_test, y_test, self._model)

        elif self._regType == RegType.ElasticNet:
            r2 = self._model.score(x_test, y_test)
            mse = mean_squared_error(y_test, y_hat)
            print 'R2:', r2
            print u'均方误差：', mse

            self._plot(x_test, y_test, y_hat, xlabel=u'样本编号', ylabel=u'房屋价格', title=self._plotTitle)
        return



def mainAdvertising():
    data = loadAdvertisingData(_pathAdvertisingData)
    plotAdvertisingData()
    y, x = data.labelAndFeature

    regressor = Regressor(y=y, x=x, regType=RegType.LinearReg)
    regressor.regression()

    return

def mainIris():
    data = loadIrisData(_pathIrisData, onlyUseTwoFeat=True)
    y, x = data.labelAndFeature

    regressor = Regressor(y=y, x=x, regType=RegType.LogisticReg)
    regressor.regression()

    return


def mainBoston():
    data = loadBostonData(_pathHousingData)
    y, x = data.labelAndFeature

    regressor = Regressor(y=y, x=x, regType=RegType.ElasticNet, plotTitle=u'波士顿房价预测')
    regressor.regression()

if __name__ == "__main__":
    mainBoston()