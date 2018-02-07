# -*-coding:utf-8-*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics

if __name__ == '__main__':
    y = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.5, 0.3, 0.8])
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    fpr = np.insert(fpr, 0, 0)
    tpr = np.insert(tpr, 0, 0)
    print(fpr)
    print(tpr)
    print(thresholds)
    auc = metrics.auc(fpr, tpr)
    print(metrics.roc_auc_score(y, y_pred))

    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(fpr, tpr, marker='o', lw=2, ls='-', mfc='g', mec='g', color='r')
    plt.plot([0, 1], [0, 1], lw=2, ls='--', c='b')
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.grid(b=True, ls='dotted')
    plt.title(u'ROC曲线', fontsize=18)
    plt.show()
