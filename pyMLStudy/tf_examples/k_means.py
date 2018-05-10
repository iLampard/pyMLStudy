import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
from tensorflow.examples.tutorials.mnist import input_data


def load_data():
    return input_data.read_data_sets('', one_hot=True)


def run(mnist):
    # Parameters# Param
    num_steps = 50  # Total steps to train
    batch_size = 1024  # The number of samples per batch
    k = 25  # The number of clusters
    num_classes = 10  # The 10 digits
    num_features = 784  # Each image is 28x28 pixels

    data_x = mnist.train.images

    X = tf.placeholder('float', [None, num_features])
    y = tf.placeholder('float', [None, num_classes])

    # K-Means Parameters
    kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                    use_mini_batch=True)

    # Build KMeans graph
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, training_op) = kmeans.training_graph()

    # each point -> centroids
    cluster_idx = cluster_idx[0]
    avg_distance = tf.reduce_mean(scores)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init, feed_dict={X: data_x})
    sess.run(init_op, feed_dict={X: data_x})

    for i in range(num_steps):
        _, d, idx = sess.run([training_op, avg_distance, cluster_idx], feed_dict={X: data_x})
        if i % 10 == 0:
            print('Step {0}: Avg Distance {1}'.format(i, d))

    counts = np.zeros(shape=(k, num_classes))

    for i in range(len(idx)):
        counts[idx[i]] += mnist.train.labels[i]

    # centroids id -> class
    lables_map = [np.argmax(c) for c in counts]
    labels_map = tf.convert_to_tensor(lables_map)

    # each point -> label
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)

    accuracy = tf.equal(cluster_label, tf.cast(tf.argmax(y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    # Test Model
    test_x, test_y = mnist.test.images, mnist.test.labels
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, y: test_y}))

    return


if __name__ == '__main__':
    mnist = load_data()
    run(mnist)
