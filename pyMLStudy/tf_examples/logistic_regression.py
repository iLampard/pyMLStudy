import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_data():
    return input_data.read_data_sets('', one_hot=True)


def run(mnist):
    # Parameters# Param
    learning_rate = 0.01
    training_epochs = 25
    batch_size = 100
    display_step = 1

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28 * 28 = 784
    y = tf.placeholder(tf.float32, [None, 10])  # 10 classes

    # model weight
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)

    # cost
    cost = tf.reduce_mean(- tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # training the model
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = mnist.train.num_examples // batch_size
            for batch in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                avg_cost += c / total_batch
            print("Epoch: {0}, cost = {1}".format(epoch + 1,
                                                  avg_cost))

        print('Optimization Finished')

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print("Accuracy: {0}".format(accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]})))


if __name__ == '__main__':
    mnist = load_data()
    run(mnist)
