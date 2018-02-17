# -*- coding:utf-8 -*-


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from xutils import CustomLogger

logger = CustomLogger('DecisionTree', log_file='DecisionTree.log')

if __name__ == '__main__':
    iris = load_iris()

    # 为了可视化，仅使用前两列特征
    x = iris.data[:, :2]
    y = iris.target

    train_features, test_features, train_labels, test_labels = train_test_split(x,
                                                                                y,
                                                                                test_size=0.33)

    # min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
    # min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则完成分支；否则，不进行分支
    model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    model.fit(train_features, train_labels)
    test_predict = model.predict(test_features)  # 测试数据

    score = accuracy_score(test_labels, test_predict)

    logger.info('Predicting accuracy is {0}'.format(score))

    N, M = 50, 50  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    y_show_hat = model.predict(x_show)  # 预测值
    # print y_show_hat.shape
    # print y_show_hat
    y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同

    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示

    plt.scatter(test_features[:, 0], test_features[:, 1], c=test_labels.ravel(), edgecolors='k', s=120, cmap=cm_dark, marker='*')  # 测试数据
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
    plt.xlabel(iris.target_names[0], fontsize=15)
    plt.ylabel(iris.target_names[1], fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title('Iris data classification using decision tree', fontsize=17)
    plt.show()
