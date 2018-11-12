import tensorflow as tf
import numpy as np


def example_1():
    x = np.random.rand(100)
    y = x * 0.1 + 0.3

    print(x.shape)

    weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    bias = tf.Variable(tf.zeros([1]))

    y_pred = weight * x + bias

    loss = tf.reduce_mean(tf.square(y - y_pred))

    optimizer = tf.train.GradientDescentOptimizer(0.5)

    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(201):
            sess.run(train)
            if step % 20 == 0:
                print(step, sess.run(weight), sess.run(bias), sess.run(loss))

    return



def example_2():
    state = tf.Variable(0, name = 'counter')
    print(state.name)

    one = tf.constant(1)
    new_val = tf.add(state, one)
    update = tf.assign(state, new_val)


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))

    return


def example_3():
    """
    placeholder 是 Tensorflow 中的占位符，暂时储存变量.
    Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).
    """
    input1 = tf.placeholder(dtype=tf.float32)
    input2 = tf.placeholder(dtype=tf.float32)

    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1: [3.], input2: [5]}))

if __name__ == '__main__':
    example_3()