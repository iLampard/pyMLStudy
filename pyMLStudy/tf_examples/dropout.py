import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


def add_layer(input, in_size, out_size, layer_name, keep_proba, activation=None):
    weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    w_plus_b = tf.add(tf.matmul(input, weight), bias)
    w_plus_b = tf.nn.dropout(w_plus_b, keep_prob=keep_proba)
    if activation is None:
        output = w_plus_b
    else:
        output = activation(w_plus_b)

    tf.summary.histogram(layer_name + '/output', output)
    return output


def main():
    digits = load_digits()
    x = digits.data
    y = digits.target
    print(y)
    print(y.shape)
    y = LabelBinarizer().fit_transform(y)
    print(y.shape)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

    xs = tf.placeholder(tf.float32, [None, 64])
    ys = tf.placeholder(tf.float32, [None, 10])

    keep_proba = tf.placeholder(tf.float32)

    l1 = add_layer(xs, 64, 50, 'l1', keep_proba=keep_proba, activation=tf.nn.tanh)
    prediction = add_layer(l1, 50, 10, 'l2', keep_proba=keep_proba, activation=tf.nn.softmax)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

    tf.summary.scalar('loss', cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test/', sess.graph)
        sess.run(init)
        for i in range(1000):
            sess.run(train_step, feed_dict={xs: train_x, ys: train_y, keep_proba: 0.5})
            if i % 50 == 0:
                train_loss = sess.run(merged, feed_dict={xs: train_x, ys: train_y, keep_proba: 0.5})
                test_loss = sess.run(merged, feed_dict={xs: test_x, ys: test_y, keep_proba: 0.5})
                train_writer.add_summary(train_loss, i)
                test_writer.add_summary(test_loss, i)

    return


if __name__ == '__main__':
    main()