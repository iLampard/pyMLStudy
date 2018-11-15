import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def gen_data(size=10000):
    """
   生成数据:
   输入数据X：在时间t，Xt的值有50%的概率为1，50%的概率为0；
   输出数据Y：在时间t，Yt的值有50%的概率为1，50%的概率为0，除此之外，如果`Xt-3 == 1`，Yt为1的概率增加50%， 如果`Xt-8 == 1`，则Yt为1的概率减少25%， 如果上述两个条件同时满足，则Yt为1的概率为75%。
   """
    x = np.array(np.random.choice(2, size=(size, 1)))
    y = []

    for i in range(size):
        thresh = 0.5
        if x[i - 3] == 1:
            thresh += 0.5
        if x[i - 8] == 1:
            thresh -= 0.25
        if np.random.rand() > thresh:
            y.append(0)
        else:
            y.append(1)

    return x, np.array(y)


def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    x = raw_x.reshape(-1, batch_size, num_steps)
    y = raw_y.reshape(-1, batch_size, num_steps)
    for i in range(x.shape[0]):
        yield (x[i], y[i])


def gen_epochs(n, batch_size, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


def main():
    batch_size = 4
    num_class = 2
    num_steps = 10
    state_size = 4
    learning_rate = 0.2
    num_epochs = 5

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='output')

    init = tf.zeros([batch_size, state_size])
    x_one_hot = tf.one_hot(x, num_class) # shape (batch_size num_steps, num_class)

    rnn_input = tf.unstack(x_one_hot, axis=1)

    cell = tf.contrib.rnn.BasicRNNCell(state_size)
    # rnn_outputs: (num_steps, batch_size, state_size)
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_input, initial_state=init)

    with tf.variable_scope('softmax'):
        w = tf.get_variable('w', [state_size, num_class])
        b = tf.get_variable('b', [num_class], initializer=tf.constant_initializer(0.0))

    logits = [tf.matmul(rnn_output, w) + b for rnn_output in rnn_outputs]
    predictions = [tf.nn.softmax(logit) for logit in logits]
    y_as_list = tf.unstack(y, num=num_steps, axis=1)
    loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for (logit, label) in
            zip(predictions, y_as_list)]

    total_loss = tf.reduce_mean(loss)

    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for index, epoch in enumerate(gen_epochs(num_epochs, batch_size, num_steps)):
            training_loss = 0
            for step, (x_, y_) in enumerate(epoch):
                loss_, total_loss_, training_state_, _ = sess.run([loss, total_loss, final_state, train_step],
                                                                  feed_dict={x: x_, y: y_})
                training_loss += total_loss_

                if step % 100 == 0 and step > 0:
                    print('Average loss at step {0} for last 100 steps: {1}'.format(step, training_loss / 100))
                    training_losses.append(training_loss / 100)
                training_loss = 0

    plt.plot(training_losses)
    plt.show()

    return


if __name__ == '__main__':
    main()
