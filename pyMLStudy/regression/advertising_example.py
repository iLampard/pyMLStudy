# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from pyMLStudy.data_processor import load_advertising_data

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

path = '..//data//Advertising.csv'


def plot_analysis():
    data = pd.read_csv(path)
    plt.figure(figsize=(9, 12))
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


def linear_regression():
    data = pd.read_csv(path)
    x = data[['TV', 'Radio']]
    y = data['Sales']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    print type(x_test)
    print x_train.shape, y_train.shape
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print model
    print linreg.coef_, linreg.intercept_

    # 为了让拟合后的数据显示成一条朝右上的曲线
    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_hat = linreg.predict(x_test)
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error

    print 'MSE = ', mse,
    print 'RMSE = ', rmse
    print 'R2 = ', linreg.score(x_train, y_train)
    print 'R2 = ', linreg.score(x_test, y_test)

    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.grid(b=True)
    plt.show()


def lasso():
    data = pd.read_csv(path)
    x = data[['TV', 'Radio']]
    y = data['Sales']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    model = Lasso()
    alpha_can = np.logspace(-3, 2, 10)
    np.set_printoptions(suppress=True)
    print 'alpha_can = ', alpha_can
    # grid search 的超参数范围是一个np array
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x_train, y_train)
    print '超参数：\n', lasso_model.best_params_

    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_hat = lasso_model.predict(x_test)
    print lasso_model.score(x_test, y_test)
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print mse, rmse

    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    linear_regression()
