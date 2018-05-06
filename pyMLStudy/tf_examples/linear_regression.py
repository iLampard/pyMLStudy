import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def run():
    # Parameters
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 50

    train_X = np.linspace(-10, 20, 100)
    train_y = train_X + np.random.randn()
    num_sample = len(train_X)

    # tf graph
    X = tf.placeholder('float')
    y = tf.placeholder('float')

    # set model weights
    W = tf.Variable(np.random.randn(), name='weight')
    b = tf.Variable(np.random.randn(), name='bias')

    # construct model
    pred = tf.add(tf.multiply(X, W), b)

    # cost
    cost = tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * num_sample)

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            for (x, y_) in zip(train_X, train_y):
                sess.run(optimizer, feed_dict={X: x, y: y_})
            if (epoch + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, y: train_y})
                print("Epoch: {0}, cost = {1}, W={2}, b={3}".format(epoch + 1,
                                                                    c,
                                                                    sess.run(W),
                                                                    sess.run(b)))
        print('Optimization finished')
        training_cost = sess.run(cost, feed_dict={X: train_X, y: train_y})
        print('Training cost = {0}, W = {1}, b = {2}'.format(training_cost,
                                                             sess.run(W),
                                                             sess.run(b)))

        # Graphic display
        plt.plot(train_X, train_y, 'ro', label='Original data')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    run()
