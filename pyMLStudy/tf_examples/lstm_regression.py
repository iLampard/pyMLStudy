import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_start = 0
num_steps = 20

def get_batch(batch_size):
    global batch_start, num_steps
    xs = np.arange(batch_start, batch_start + num_steps * batch_size).reshape((batch_size, num_steps)) / (10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)

    batch_start += num_steps

    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class LSTMRNN(object):
    def __init__(self, num_steps, input_size, output_size, cell_size, batch_size, learning_rate):
        self.num_steps = num_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, num_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, num_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('lstm_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def add_input_layer(self):
        x_in = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch_size * num_steps, input_size)
        w_in = self.weight_variable([self.input_size, self.cell_size])
        b_in = self.bias_variable([self.cell_size, ])
        with tf.name_scope('x_plus_b'):
            y_in = tf.matmul(x_in, w_in) + b_in
        self.y_in = tf.reshape(y_in, [-1, self.num_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell,
                                                                     self.y_in,
                                                                     initial_state=self.cell_init_state,
                                                                     time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        x_out = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        w_out = self.weight_variable([self.cell_size, self.output_size])
        b_out = self.bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('w_plus_b'):
            self.pred = tf.matmul(x_out, w_out) + b_out

    def weight_variable(self, shape, name='weights'):
        init = tf.random_normal_initializer(mean=0., stddev=1.)
        return tf.get_variable(shape=shape, initializer=init, name=name)

    def bias_variable(self, shape, name='biases'):
        init = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=init)

    def compute_cost(self):
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.num_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='loss'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(tf.reduce_sum(loss, name='loss_sum'),
                               self.batch_size,
                               name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))


if __name__ == '__main__':
    batch_size = 50
    input_size = 1
    output_size = 1
    cell_size = 10
    lr = 0.006
    model = LSTMRNN(num_steps, input_size, output_size, cell_size, batch_size, lr)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    plt.ion()
    plt.show()
    for i in range(100):
        seq, res, xs = get_batch(batch_size)
        if i == 0:
            feed_dict = {model.xs: seq,
                         model.ys: res}
        else:
            feed_dic = {model.xs: seq,
                        model.ys: res,
                        model.cell_init_state: state}  # use last state as the initial state for this run

        _, cost, state, pred = sess.run([model.train_op, model.cost, model.cell_final_state, model.pred],
                                        feed_dict=feed_dict)
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:num_steps], 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:
            print('cost is {0}'.format(round(cost, 4)))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)