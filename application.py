import tensorflow as tf
import tempfile
import sys
import numpy as np
from random import randint
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from matrix import Matrix
FLAGS = None


def deepnn(x):
    # Reshape input
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):

    # load MNIST data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_conv, keep_prob = deepnn(x)

    # restore session
    sess = read_model()
    tf.global_variables_initializer()

    # train(mnist, y_conv, y_, x, keep_prob)

    recognize_random(mnist, sess, y_conv, x, keep_prob)

    # test_accuracy(mnist, sess, y_conv, x, y_, keep_prob)

    # recognize_user_input(sess, y_conv, x, keep_prob)


def recognize_random(mnist, sess, y_conv, x, keep_prob):
    num = randint(0, mnist.test.images.shape[0])
    img = mnist.test.images[num]

    plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    plt.show()

    classification = sess.run(tf.argmax(y_conv, 1), feed_dict={x: [img], keep_prob: 1.0})
    print("Your digit is: ", classification[0])


def train(mnist, y_conv, y_, x, keep_prob):
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        test_accuracy(mnist, sess, y_conv, x, y_, keep_prob)
        save_model(sess)


def save_model(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, './models/mnist_deep.ckpt')
    print("Model successfully saved. Path: %s" % save_path)


def read_model():
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, './models/mnist_deep.ckpt')
    return sess


def test_accuracy(mnist, sess, y_conv, x, y_, keep_prob):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    with sess.as_default():
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def recognize_user_input(sess, y_conv, x, keep_prob):

    def classify(data):
        data = np.array(data).flatten()
        classification = sess.run(tf.argmax(y_conv, 1), feed_dict={x: [data], keep_prob: 1.0})
        print("Your digit is: ", classification[0])
        plt.imshow(data.reshape(28, 28), cmap=plt.cm.binary)
        plt.show()

    Matrix(classify)


if __name__ == '__main__':
    tf.app.run(main=main, argv=sys.argv)
