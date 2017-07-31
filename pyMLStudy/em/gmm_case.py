# -*- coding:utf-8 -*-


import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


def e_step(mu, sigma, a, y):
    phi_array = [norm.pdf(y, mu_k, sigma_k) for mu_k, sigma_k in zip(mu, sigma)]
    phi_matrix = np.vstack(phi_array).T
    denominator_array = [np.dot(a, phi_matrix[j, :]) for j in range(len(y))]
    gamma_array = [a * phi_matrix[j, :] / denominator_array[j] for j in range(len(y))]
    gamma_matrix = np.vstack(gamma_array)
    return gamma_matrix


def m_step(gamma_matrix, y):
    nb_model = gamma_matrix.shape[1]
    mu = [np.dot(gamma_matrix[:, k], y) / np.sum(gamma_matrix[:, k]) for k in range(nb_model)]
    sigma = [np.dot(gamma_matrix[:, k], (y - mu[k]) ** 2) / np.sum(gamma_matrix[:, k]) for k in range(nb_model)]
    a = [np.sum(gamma_matrix[:, k]) / len(y) for k in range(nb_model)]
    return mu, sigma, a


def update_params(params, y):
    mu, sigma, a = params
    gamma_matrix = e_step(mu, sigma, a, y)
    mu, sigma, a = m_step(gamma_matrix, y)
    return np.array([mu, sigma, a])


def em(init_params, y, tol=10 ** -5):
    prev_params = init_params
    current_params = update_params(prev_params, y)
    while np.max(np.abs(current_params - prev_params)) > tol:
        prev_params = current_params
        current_params = update_params(prev_params, y)
        print current_params[0]
    return current_params


def brutal_em(y):
    num_iter = 100
    n = len(y)
    d = 1
    # 随机指定
    # mu1 = np.random.standard_normal(d)
    # print mu1
    # mu2 = np.random.standard_normal(d)
    # print mu2
    mu1 = np.min(y)
    mu2 = np.max(y)
    sigma1 = np.std(y)
    sigma2 = np.std(y)
    pi = 0.5
    # EM
    for i in range(num_iter):
        # E Step
        tau1 = pi * norm.pdf(y, mu1, sigma1)
        tau2 = (1 - pi) * norm.pdf(y, mu2, sigma2)
        gamma = tau1 / (tau1 + tau2)
        # M Step
        mu1 = np.dot(gamma, y) / np.sum(gamma)
        mu2 = np.dot((1 - gamma), y) / np.sum((1 - gamma))
        sigma1 = np.dot(gamma * (y - mu1).T, y - mu1) / np.sum(gamma)
        sigma2 = np.dot((1 - gamma) * (y - mu2).T, y - mu2) / np.sum(1 - gamma)
        pi = np.sum(gamma) / n
        print i, ":\t", mu1, mu2
    print '类别概率:\t', pi
    print '均值:\t', mu1, mu2
    print '方差:\n', sigma1, '\n\n', sigma2, '\n'


def sklearn_em(y):
    g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
    print y_test.shape
    g.fit(y_test[np.newaxis].T)
    print '类别概率:\t', g.weights_[0]
    print '均值:\n', g.means_, '\n'
    print '方差:\n', g.covariances_, '\n'
    mu1, mu2 = g.means_
    sigma1, sigma2 = g.covariances_
    return


if __name__ == "__main__":
    y_test = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75])
    m = np.mean(y_test)
    v = np.std(y_test)
    init_params_test = ([np.min(y_test), np.max(y_test)], [v, v], [0.6, 0.4])
    em(init_params_test, y_test)
    brutal_em(y_test)
