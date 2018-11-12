import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, n_layer, activation=None):
    layer_name = 'layer_{0}'.format(n_layer)

    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weight = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weight', weight)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='bias')
            tf.summary.histogram(layer_name + '/bias', bias)
        with tf.name_scope('w_plus_b'):
            w_plus_b = tf.add(tf.matmul(inputs, weight), bias)
        if activation is None:
            output = w_plus_b
        else:
            output = activation(w_plus_b)

    return output


def main():
    x = np.linspace(-1, 1, 300).reshape(-1, 1)
    noise = np.random.normal(0, 0.05, x.shape)
    y = np.square(x) - 0.5 + noise

    with tf.name_scope('input'):
        xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
        ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

    l1 = add_layer(xs, 1, 10, 1, activation=tf.nn.relu)
    prediction = add_layer(l1, 10, 1, 2, activation=None)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(init)
        for i in range(1000):
            sess.run(train_step, feed_dict={xs: x, ys: y})
            if i % 50 == 0:
                result = sess.run(merged, feed_dict={xs: x, ys: y})
                writer.add_summary(result, i)


    return


if __name__ == '__main__':
    main()
