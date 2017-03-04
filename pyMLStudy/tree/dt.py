#-*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from pyMLStudy.dataProcessor import loadIrisData


mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False



_pathIrisData = '..//data//Iris.data'


def mainIris():
    data = loadIrisData(_pathIrisData, onlyUseTwoFeat=True)
    y, x = data.labelAndFeature

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    print y_test

    # 决策树参数估计
    # min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
    # min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则完成分支；否则，不进行分支
    model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    model = model.fit(x_train, y_train)
    y_hat = model.predict(x_test)      # 测试数据

    y_test = y_test.reshape(-1)
    print y_hat
    print y_test
    result = (y_hat == y_test)   # True则预测正确，False则预测错误
    acc = np.mean(result)
    print '准确度: %.2f%%' % (100 * acc)

    # 过拟合：错误率
    depth = np.arange(1, 15)
    err_list = []
    for d in depth:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        clf = clf.fit(x_train, y_train)
        y_test_hat = clf.predict(x_test)  # 测试数据
        result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
        err = 1 - np.mean(result)
        err_list.append(err)
        # print d, ' 准确度: %.2f%%' % (100 * err)
        print d, ' 错误率: %.2f%%' % (100 * err)
    plt.figure(facecolor='w')
    plt.plot(depth, err_list, 'ro-', lw=2)
    plt.xlabel(u'决策树深度', fontsize=15)
    plt.ylabel(u'错误率', fontsize=15)
    plt.title(u'决策树深度与过拟合', fontsize=17)
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    mainIris()