# https://blog.csdn.net/u013082989/article/details/73469095

import numpy as np
import tensorflow as tf


def gen_sample(size=1000000):
    x = np.array(np.random.choice(2, size=(size,)))
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
    data_length = len(raw_x)

    # 首先将数据切分成batch_size份，0-batch_size，batch_size-2*batch_size。。。
    batch_partition_length = data_length // batch_size  # ->5000
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)  # ->(200, 5000)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)  # ->(200, 5000)

    for i in range(batch_size):
        data_x[i] = raw_x[i * batch_size, (i + 1) * batch_size]
        data_y[i] = raw_y[i * batch_size, (i + 1) * batch_size]

    # 因为RNN模型一次只处理num_steps个数据，所以将每个batch_size在进行切分成epoch_size份，
    # 每份num_steps个数据。注意这里的epoch_size和模型训练过程中的epoch不同。
    epoch_size = batch_partition_length // num_steps

    # x是0-num_steps， batch_partition_length -batch_partition_length +num_steps。。。共batch_size个
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]  # ->(200, 5)
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


# n就是训练过程中用的epoch，即在样本规模上循环的次数
def gen_epochs(n, batch_size, num_steps):
    for i in range(n):
        yield gen_batch(gen_sample(), batch_size, num_steps)


# 使之定义为reuse模式，循环使用，保持参数相同
def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    # 定义rnn_cell具体的操作，这里使用的是最简单的rnn，不是LSTM
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)


if __name__ == '__main__':
    '''超参数'''
    num_steps = 5
    batch_size = 200
    num_classes = 2
    state_size = 4
    learning_rate = 0.1

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

    # #RNN的初始化状态，全设为零。注意state是与input保持一致，接下来会有concat操作，所以这里要有batch的维度。即每个样本都要有隐层状态
    init_state = tf.zeros([batch_size, state_size])

    # 将输入转化为one-hot编码，两个类别。[batch_size, num_steps, num_classes]
    x_one_hot = tf.one_hot(x, num_classes)

    # 将输入unstack，即在num_steps上解绑，方便给每个循环单元输入。这里可以看出RNN每个cell都处理一个batch的输入（即batch个二进制样本输入）
    rnn_inputs = tf.unstack(x_one_hot, axis=1)

    # 定义rnn_cell的权重参数，
    with tf.variable_scope('rnn_cell'):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

    state = init_state
    rnn_outputs = []

    # 循环num_steps次，即将一个序列输入RNN模型
    for rnn_input in rnn_inputs:
        state = rnn_cell(rnn_input, state)
        rnn_outputs.append(state)

    final_state = rnn_outputs[-1]

    # 定义softmax层
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    # 注意，这里要将num_steps个输出全部分别进行计算其输出，然后使用softmax预测
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    predictions = [tf.nn.softmax(logit) for logit in logits]

    # Turn our y placeholder into a list of labels
    y_as_list = tf.unstack(y, num=num_steps, axis=1)

    # losses and train_step
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
              logit, label in zip(logits, y_as_list)]
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)
