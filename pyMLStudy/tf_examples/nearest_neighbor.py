import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_data():
    return input_data.read_data_sets('', one_hot=True)


def run(mnist):
    train_x, train_y = mnist.train.next_batch(5000)
    test_x, test_y = mnist.train.next_batch(200)

    train_holder = tf.placeholder('float', [None, 784])
    test_holder = tf.placeholder('float', [784])

    # l1 distance
    distance = tf.reduce_sum(tf.abs(tf.add(train_holder, tf.negative(test_holder))), reduction_indices=1)

    pred = tf.argmin(distance, 0)

    init = tf.global_variables_initializer()

    accuracy = 0.0
    with tf.Session() as sess:
        sess.run(init)
        for i in range(len(test_x)):
            index = sess.run(pred, feed_dict={train_holder: train_x, test_holder: test_x[i, :]})
            pred_test = np.argmax(train_y[index])
            act_test = np.argmax(test_y[i])
            if pred_test == act_test:
                accuracy += 1.0
            print('Test {0}: prediction = {1},  true class = {2}'.format(i,
                                                                         pred_test,
                                                                         act_test))
        print('Accuracy = {0}'.format(accuracy / len(test_x)))


if __name__ == '__main__':
    mnist = load_data()
    run(mnist)
