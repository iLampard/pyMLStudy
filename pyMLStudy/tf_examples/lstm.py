import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def rnn(x, weight, bias, batch_size, num_inputs, num_steps, num_hidden):
    x = tf.reshape(x, [-1, num_inputs])
    x_in = tf.matmul(x, weight['in']) + bias['in']
    x_in = tf.reshape(x_in, [-1, num_steps, num_hidden])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)

    # final_states[1] 就是短时记忆h
    results = tf.matmul(final_states[1], weight['out']) + bias['out']

    return results


def main():
    lr = 0.001
    num_iters = 10000
    batch_size = 128

    num_inputs = 28
    num_steps = 28
    num_hidden = 128
    num_class = 10

    xs = tf.placeholder(tf.float32, [None, num_steps, num_inputs])
    ys = tf.placeholder(tf.float32, [None, 10])

    weights = {'in': tf.Variable(tf.random_normal([num_inputs, num_hidden])),
               'out': tf.Variable(tf.random_normal([num_hidden, num_class]))}

    bias = {'in': tf.Variable(tf.random_normal([num_hidden])),
            'out': tf.Variable(tf.random_normal([num_class]))}

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    prediction = rnn(xs, weights, bias, batch_size, num_inputs, num_steps, num_hidden)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=ys))
    train_step = tf.train.AdadeltaOptimizer(lr).minimize(cross_entropy)

    correct_pred = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(ys, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step * batch_size < num_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, num_steps, num_inputs])
            sess.run(train_step, feed_dict={xs: batch_xs,
                                            ys: batch_ys})
            if step % 20 == 0:
                print(sess.run(accuracy, feed_dict={xs: batch_xs,
                                                    ys: batch_ys}))
        step += 1
    return


if __name__ == '__main__':
    main()
