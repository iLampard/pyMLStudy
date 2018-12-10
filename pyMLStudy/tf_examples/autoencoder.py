import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# build the encoder
def encode(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decode(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


def main():
    lr = 0.01
    training_epochs = 5
    batch_size = 256
    display_step = 1
    num_input = 784  # MNIST data input (img shape: 28*28)
    num_hidden_1 = 256
    num_hidden_2 = 128

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input]))
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([num_input]))
    }

    mnist = input_data.read_data_sets('./mnist', one_hot=False)  # use not one-hotted target data

    x = tf.placeholder(tf.float32, [None, 784])
    encoder_op = encode(x, weights, biases)
    decoder_op = decode(encoder_op, weights, biases)

    y_pred = decoder_op
    y_true = x
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int(mnist.train.num_examples / batch_size)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    return


if __name__ == '__main__':
    main()
