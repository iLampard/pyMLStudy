# -*- coding:utf-8 -*-


import numpy as np
from scipy.stats import norm
from itertools import product


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
    return current_params


if __name__ == "__main__":
    y_test = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75])
    m = np.mean(y_test)
    v = np.std(y_test)
    print m, v
    init_params_test = ([m, m], [v, v], [0.6, 0.4])
    print em(init_params_test, y_test)
