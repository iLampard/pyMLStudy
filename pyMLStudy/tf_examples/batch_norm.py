import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ACTIVATION = tf.nn.relu
NUM_LAYERS = 7
NUM_HIDDEN_UNITS = 30


# https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-13-BN/

def fix_seed(seed=1):
    # reproducible
    np.random.seed(seed)
    tf.set_random_seed(seed)


def plot_hist(inputs, inputs_norm):
    # plot histogram for the inputs of every layer
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j * len(all_inputs) + (i + 1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
    plt.draw()
    plt.pause(0.01)


def build_net(xs, ys, norm):
    def add_layer(input, in_size, out_size, activation_function=None, norm=False):
        weight = tf.Variable(tf.random_normal([in_size, out_size],
                                              mean=0, stddev=1.0))
        bias = tf.Variable(tf.zeros([out_size]) + 0.1)
        w_plus_b = tf.matmul(input, weight) + bias

        if norm:
            fc_mean, fc_var = tf.nn.moments(w_plus_b, axes=[0])
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            eps = 0.001
            w_plus_b = tf.nn.batch_normalization(w_plus_b, fc_mean, fc_var, shift, scale, eps)
            # 上面那一步, 在做如下事情:
            # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
            # Wx_plus_b = Wx_plus_b * scale + shift

        if activation_function is None:
            output = w_plus_b
        else:
            output = activation_function(w_plus_b)
        return output

    fix_seed(1)

    layers_input = [xs]

    # loop 建立所有层
    for l_n in range(NUM_LAYERS):
        layer_input = layers_input[l_n]
        in_size = layers_input[l_n].get_shape()[1].value

        output = add_layer(layer_input,
                           in_size,
                           NUM_HIDDEN_UNITS,
                           ACTIVATION,
                           norm=norm)
        layers_input.append(output)

    prediction = add_layer(layers_input[-1], 30, 1, activation_function=None)
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    return [train_op, cost, layers_input]


def main():
    x = np.linspace(-7, 10, 500)[:, np.newaxis]
    noise = np.random.normal(0, 8, x.shape)
    y = np.square(x) - 5 + noise
    # plt.scatter(x, y)
    # plt.show()

    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    train_op, cost, layers_inputs = build_net(xs, ys, norm=False)  # without BN
    train_op_norm, cost_norm, layers_inputs_norm = build_net(xs, ys, norm=True)  # with BN

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 记录两种网络的 cost 变化
    cost_his = []
    cost_his_norm = []
    record_step = 5
    plt.ion()
    plt.figure(figsize=(7, 3))

    for i in range(251):
        if i % 50 == 0:
            all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm], feed_dict={xs: x, ys: y})
            plot_hist(all_inputs, all_inputs_norm)

        sess.run(train_op, feed_dict={xs: x, ys: y})
        sess.run(train_op_norm, feed_dict={xs: x, ys: y})

        if i % record_step == 0:
            cost_his.append(sess.run(cost, feed_dict={xs: x, ys: y}))
            cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x, ys: y}))

    plt.ioff()
    plt.figure()
    plt.plot(np.arange(len(cost_his)) * record_step, np.array(cost_his), label='no BN')  # no norm
    plt.plot(np.arange(len(cost_his)) * record_step, np.array(cost_his_norm), label='BN')  # norm
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    main()
