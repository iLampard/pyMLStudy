# -*- coding:utf-8 -*-

from numpy import *


# y = mx + b
# m is slope, b is y-intercept

def MSE(weight, x, y):
    error = mat(x) * mat(weight).T - y
    ret = sum(error.T * error) / float(len(x))
    return ret


def update_weight(weight, x, y, learning_rate):
    gradient = calc_gradient_matrix(weight, x, y)
    weight = weight - gradient * learning_rate
    return weight


def calc_gradient_matrix(weight, x, y):
    X = mat(x)
    gradient = X.T * X * weight - X.T * y
    gradient /= float(len(x))
    return gradient


def calc_gradient(b_current, m_current, x, y):
    weight = array([b_current, m_current]).reshape(-1, 1)
    x = hstack((ones(len(x)).reshape(-1, 1), x.reshape(-1, 1)))
    y = y.reshape(-1, 1)
    gradient = calc_gradient_matrix(weight, x, y)
    b_gradient = gradient[0, 0]
    m_gradient = gradient[1, 0]
    return b_gradient, m_gradient


def step_gradient(b_current, m_current, points, learningRate):
    b_gradient, m_gradient = calc_gradient(b_current, m_current, x=points[:, 0], y=points[:, 1])
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.00001
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    num_iterations = 1000
    x = hstack((ones(len(points)).reshape(-1, 1), points[:, 0].reshape(-1, 1)))
    y = points[:, 1].reshape(-1, 1)
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}". \
        format(initial_b, initial_m, MSE([initial_b, initial_m], x, y))
    print "Running..."
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
                                                                      MSE([b, m], x, y))


if __name__ == '__main__':
    run()
