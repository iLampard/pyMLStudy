# -*- coding:utf-8 -*-

import matplotlib as mpl
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pyMLStudy.sklearn_examples.data_processor import load_bipartition
from pyMLStudy.sklearn_examples.data_processor import load_iris_data

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

_pathIrisData = '..//data//Iris.data'
_pathBipartition = '..//data//bipartition.csv'


def show_accuracy(a, b, tip):
    print tip + '正确率：', float(len(a[a == b])) / len(a)


def bipartition_main():
    data = load_bipartition(_pathBipartition)
    y, x = data.label_feature

    # 分类器
    clf_param = (('linear', 0.1), ('linear', 0.5), ('linear', 1), ('linear', 2),
                 ('rbf', 1, 0.1), ('rbf', 1, 1), ('rbf', 1, 10), ('rbf', 1, 100),
                 ('rbf', 5, 0.1), ('rbf', 5, 1), ('rbf', 5, 10), ('rbf', 5, 100))

    for i, param in enumerate(clf_param):
        clf = svm.SVC(C=param[1], kernel=param[0])
        if param[0] == 'rbf':
            clf.gamma = param[2]
            title = u'高斯核，C=%.1f，$\gamma$ =%.1f' % (param[1], param[2])
        else:
            title = u'线性核，C=%.1f' % param[1]

        clf.fit(x, y)
        y_hat = clf.predict(x)
        print 'accurity is {0}'.format(accuracy_score(y_hat, y))  # 准确率
        print title
        print '支撑向量的数目：', clf.n_support_
        print '支撑向量的系数：', clf.dual_coef_
        print '支撑向量：', clf.support_


def svm_classification():
    data = load_iris_data(_pathIrisData, onlyUseTwoFeat=False)
    y, x = data.label_feature

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train)

    # 准确率
    print clf.score(x_train, y_train)  # 精度
    y_hat = clf.predict(x_train)
    show_accuracy(y_hat, y_train, '训练集')
    print clf.score(x_test, y_test)
    y_hat = clf.predict(x_test)
    show_accuracy(y_hat, y_test, '测试集')
    print accuracy_score(y_test, y_hat)

    return


if __name__ == "__main__":
    bipartition_main()
