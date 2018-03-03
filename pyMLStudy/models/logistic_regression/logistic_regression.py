# -*- encoding:utf-8 -*-


import numpy as np
from xutils import CustomLogger
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from argcheck import expect_types
import matplotlib.pyplot as plt

logger = CustomLogger('LogisticRegression', log_file='LogisticRegression.log')


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class LogisticReg(object):
    """
    gradient = sum ( y_i - proba_i) x_i

    """

    def __init__(self, max_iteration=5000, learning_rate=0.0001):
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate
        self.weight = None

    @expect_types(x=np.ndarray, y=np.ndarray)
    def fit(self, x, y, method='gd', eps=10e-2):
        logger.info('Do the calibration using {0} method'.format(method))
        num_data, num_feature = x.shape
        labels = np.mat(y).transpose()

        # 加一列常数列
        X = np.mat(np.concatenate((x, np.ones((num_data, 1))), axis=1))
        w = np.zeros(num_feature + 1).reshape(-1, 1)
        for iteration in range(self.max_iteration):
            error = labels - sigmoid(X * w)
            error_sum = error.T * error
            w += self.learning_rate * X.T * error
            if error_sum[0, 0] < eps:
                break
            logger.info('Iteration {0}, error {1}'.format(iteration,
                                                          error_sum[0, 0]))
        self.weight = w
        logger.info('Calibration finished')

    @expect_types(x=np.ndarray)
    def predict(self, x):
        if self.weight is not None:
            num_data, num_feature = x.shape
            X = np.mat(np.concatenate((x, np.ones((num_data, 1))), axis=1))
            proba = sigmoid(X * self.weight)
            labels = np.asarray(proba).ravel()
            labels[labels > 0.5] = 1
            labels[labels <= 0.5] = 0
            return labels


def plot_fit(weights, test_features, pred_labels):

    # 第一类
    xcord1 = []
    ycord1 = []
    # 第二类
    xcord2 = []
    ycord2 = []

    num_data = len(test_features)
    for i in range(num_data):
        if pred_labels[i] == 0:
            xcord1.append(test_features[i][0])
            ycord1.append(test_features[i][1])
        else:
            xcord2.append(test_features[i][0])
            ycord2.append(test_features[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    x_min, x_max = test_features[:, 0].min() - .5, test_features[:, 0].max() + .5
    x = np.arange(x_min, x_max, 0.1)
    y = (-weights[2] - weights[0] * x) / weights[1]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Logsitic Regression')
    plt.show()
    pass


if __name__ == '__main__':
    x, labels = make_moons(500, noise=0.2)
    train_features, test_features, train_labels, test_labels = train_test_split(x,
                                                                                labels,
                                                                                test_size=0.33)
    logger.info('Start Training...')
    model = LogisticReg()
    model.fit(train_features, train_labels)
    logger.info('Traning finished...')

    logger.info('Start Predicting...')
    test_predict = model.predict(test_features)
    score = accuracy_score(test_labels, test_predict)
    logger.info('Predicting accuracy is {0}'.format(score))

    plot_fit(model.weight, test_features, test_predict)
