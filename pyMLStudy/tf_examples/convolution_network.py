import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    # 本层我们的卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # 因为采用了SAME的padding方式，输出图片的大小没有变化依然是28x28，只是厚度变厚了，因此现在的输出大小就变成了28x28x32
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)  # shape(14, 14, 64)
    h_pool2 = max_pool(h_conv2)  # shape(7, 7, 64)

    # 全连接层
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = weight_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)  # 此时已不是卷积层
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([1024, 10])
    b_fc2 = weight_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run([train_step, accuracy], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            print(sess.run([cross_entropy, accuracy], feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1}))

    return


if __name__ == '__main__':
    main()
