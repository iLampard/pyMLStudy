import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ACTIVATION = tf.nn.relu
NUM_LAYERS = 7
NUM_HIDDEN_UNITS = 30

# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-13-BN/

def fix_seed(seed=1):
    # reproducible
    np.random.seed(seed)
    tf.set_random_seed(seed)


def build_net(xs, ys, norm):
    def add_layer(input, in_size, out_size, activation_function=None, norm=False):
        weight = tf.Variable(tf.random_normal([in_size, out_size],
                                              mean=0, stddev=1.0))
        bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        w_plus_b = tf.matmul(input, weight) + bias
        if activation_function is None:
            output = w_plus_b
        else:
            output = activation_function(w_plus_b)
        return output

    fix_seed(1)

    layers_input = [xs]

    # loop 建立所有层
    for l_n in range(NUM_LAYERS):
        layer_input = layers_input[l_n]
        in_size = layers_input[l_n].get_shape()[1].value

        output = add_layer(layer_input,
                           in_size,
                           NUM_HIDDEN_UNITS,
                           ACTIVATION)
        layers_input.append(output)

    prediction = add_layer(layers_input[-1], 30, 1, activation_function=None)
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    return [train_op, cost, layers_input]


def main():
    x = np.linspace(-7, 10, 500)[:, np.newaxis]
    noise = np.random.normal(0, 8, x.shape)
    y = np.square(x) - 5 + noise
    # plt.scatter(x, y)
    # plt.show()

    return
