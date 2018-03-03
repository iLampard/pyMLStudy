# -*- coding:utf-8 -*-


import numpy as np


def mu_j(pi, p, q, y_j):
    nominator = pi * p ** y_j * (1 - p) ** (1 - y_j)
    denominator = pi * p ** y_j * (1 - p) ** (1 - y_j) + (1-pi) * q ** y_j * (1 - q) ** (1 - y_j)
    return nominator / denominator


def e_step(pi, p, q, y):
    ret = [mu_j(pi, p, q, y_j) for y_j in y]
    return np.array(ret)


def m_step(mu, y):
    pi = np.mean(mu)
    p = np.dot(mu, y) / (pi * len(y))
    q = (np.sum(y) - np.dot(mu, y)) / (len(y) - np.sum(mu))
    return pi, p, q


def update_params(params, y):
    pi, p, q = params
    mu = e_step(pi, p, q, y)
    pi, p, q = m_step(mu, y)
    return np.array([pi, p, q])


def em(init_params, y, tol=10 ** -10):
    prev_params = init_params
    current_params = update_params(prev_params, y)
    while np.max(np.abs(current_params - prev_params)) > tol:
        prev_params = current_params
        current_params = update_params(prev_params, y)
    return current_params


if __name__ == "__main__":
    y_test = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    init_params_test = np.array([0.46, 0.55, 0.67])
    print em(init_params_test, y_test)
