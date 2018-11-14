import tensorflow as tf
import numpy as np


def main():
    sess = tf.Session()

    y_hat = tf.convert_to_tensor(np.array([[0.5, 1.5, 0.1], [2.2, 1.3, 1.7]]))
    print(sess.run(y_hat))

    # 归一化
    y_hat_softmax = tf.nn.softmax(y_hat)
    print(sess.run(y_hat_softmax))

    y_true = tf.convert_to_tensor(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    loss_1 = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_hat_softmax), reduction_indices=[1]))
    print(sess.run(loss_1))
    loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=y_true))
    print(sess.run(loss_2))

    return


if __name__ == '__main__':
    main()
