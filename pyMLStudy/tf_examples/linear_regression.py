import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


try:
    from keras.models import Sequential
    from keras.layers import Dense
except ImportError:
    pass

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



def run_multiple_regression():
    # Parameters
    learning_rate = 0.01
    training_epochs = 100
    display_step = 50

    train_X_1 = np.linspace(-10, 20, 100).reshape(-1, 1)
    train_X_2 = np.linspace(-10, 20, 100).reshape(-1, 1)
    train_X = np.concatenate((train_X_1, train_X_2), axis=1)
    train_y = train_X_1 + train_X_2 + np.random.randn()
    num_sample, num_features = train_X.shape

    # tf graph
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    # set model weights
    W = tf.Variable(np.random.randn(num_features, 1).astype(np.float32), name='weight')
    b = tf.Variable(tf.zeros([1]), name='bias')

    # construct model
    pred = tf.add(tf.matmul(X, W), b)

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
                sess.run(optimizer, feed_dict={X: x.reshape(1, -1), y: y_.reshape(1, -1)})
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




def run_multiple_regression_keras():
    train_X_1 = np.linspace(-10, 20, 100).reshape(-1, 1)
    train_X_2 = np.linspace(-10, 20, 100).reshape(-1, 1)
    train_X = np.concatenate((train_X_1, train_X_2), axis=1)
    train_y = train_X_1 + train_X_2 + np.random.randn()
    num_sample, num_features = train_X.shape

    output_dim = 1
    input_dim = num_features
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, kernel_initializer='normal', activation='linear'))
    model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_X, train_y, epochs=2000, verbose=1)

    # Print computed weight values
    layer = model.layers
    w, b = layer[0].get_weights()
    print('Weights: w1 = {:.1f}, w2 = {:.1f}, b = {:.1f}'.format(w[0][0], w[1][0], b[0]))
    print('errors = {}'.format(abs(model.predict(train_X) - train_y)))
    return



if __name__ == '__main__':
    run_multiple_regression_keras()
