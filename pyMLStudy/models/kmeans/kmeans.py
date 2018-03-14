# -*- encoding:utf-8 -*-

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 一个样本与数据集中所有样本的欧式距离的平方
def euclidean_distance(one_sample, X):
    num_sample = X.shape[0]
    distance = np.power(np.tile(one_sample, (num_sample, 1)) - X, 2).sum(axis=1)
    return distance


class KMeans(object):
    def __init__(self, k, max_iter=1000, tol=10e-3):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    # 初始化质心 - 随机取k个样本
    def init_centroids(self, X):
        """
        :param X: 样本数据集
        :return: np.array, k * num_feature
        """
        num_sample, num_feature = np.shape(X)
        centroids = np.zeros((self.k, num_feature))
        for i in range(self.k):
            centroids[i] = X[np.random.choice(range(num_sample))]
        return centroids

    def update_centroids(self, cluster, X):
        num_sample, num_feature = np.shape(X)
        centroids = np.zeros((self.k, num_feature))
        for center in range(self.k):
            sub_cluster = X[np.nonzero(cluster[:, 0] == center)[0]]
            centroids[center, :] = np.mean(sub_cluster, axis=0)
        return centroids

    def fit(self, X):
        num_sample = X.shape[0]

        # cluster 有两列，第一列是质心的索引，第二列是点到质心的距离
        cluster = np.zeros((X.shape[0], 2))

        # 初始化质心
        centroids = self.init_centroids(X)

        for _ in range(self.max_iter):
            # 计算样本到各个质心的距离
            for i in range(num_sample):
                distance = euclidean_distance(X[i], centroids)

                # 找到距离样本点最近的质心
                min_index = np.argmin(distance)

                # 更新cluster
                cluster[i] = np.array([min_index, distance[min_index]])

            prev_controids = centroids

            # 更新centroids
            centroids = self.update_centroids(cluster, X)

            # 判断质心是否有变动
            if abs((centroids - prev_controids).any()) < self.tol:
                break

        return centroids, cluster

    def predict(self, X):
        _, cluster = self.fit(X)
        return cluster[:, 0]


if __name__ == '__main__':
    X, y = make_blobs(n_samples=5000,
                      n_features=3,
                      centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
                      cluster_std=[0.2, 0.1, 0.2, 0.2],
                      random_state=9)

    clf = KMeans(k=4)
    y_pred = clf.predict(X)

    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)

    y = y_pred
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2], c='g')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2], c='b')
    plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], X[y == 2][:, 2], c='k')
    plt.scatter(X[y == 3][:, 0], X[y == 3][:, 1], X[y == 3][:, 2], c='y')
    plt.show()
