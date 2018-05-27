# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """

    o_h = np.zeros(n)
    o_h[x] = 1.
    return tf.convert_to_tensor(o_h, dtype=tf.float32)


num_classes = 4
batch_size = 10

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(i, num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 200, 200)
        image = tf.reshape(image, [200, 200, 3])
        image = tf.to_float(image) / 255. - 0.5 # Normalizacion ???
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue, allow_smaller_final_batch = True)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False): #La entrada de la red es el batch entero??
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=5, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        o5 = tf.layers.conv2d(inputs=o4, filters=128, kernel_size=3, activation=tf.nn.relu)
        o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)
        o7 = tf.layers.conv2d(inputs=o6, filters=356, kernel_size=2, activation=tf.nn.relu)
        o8 = tf.layers.max_pooling2d(inputs=o7, pool_size=2, strides=2)
        hf = tf.layers.flatten(o8)
        h1 = tf.layers.dense(inputs=hf, units=128, activation=tf.nn.relu)
        h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu)
        h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h2, units=4, activation=tf.nn.softmax)
    return y

example_batch_train, label_batch_train = dataSource(["train/0/*.jpg", "train/1/*.jpg", "train/2/*.jpg", "train/3/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["valid/0/*.jpg", "valid/1/*.jpg", "valid/2/*.jpg", "valid/3/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["test/0/*.jpg", "test/1/*.jpg", "test/2/*.jpg", "test/3/*.jpg"], batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - label_batch_valid))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:

    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    error = 0
    perror = 0
    epoch = 0
    epochs = []
    errors = []

    #while abs(perror - error) >= perror * 0.000001:
    for _ in range (500):
        n, c = sess.run([optimizer, cost])
        if epoch % 20 == 0:
            print("Epoch:", epoch, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(tf.argmax(example_batch_valid_predicted,1)))
            print("Error Valid Set:", sess.run(cost_valid))
            print("Error Training Set:", c)
        perror = error
        error= sess.run(cost_valid)
        errors.append(error)
        epochs.append(epoch)
        epoch += 1

    plt.plot(epochs, errors)
    plt.show()

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    results = []
    labels = []

    for _ in range(15):
        batch_result = sess.run(example_batch_test_predicted)
        batch_label = sess.run(label_batch_test)
        results.extend(batch_result)
        labels.extend(batch_label)
    mistakes = 0

    for b, r in zip(labels, results):
        if np.argmax(b) != np.argmax(r):
            mistakes = mistakes + 1
            print(b, "-->", r)
    print ("Mistakes:", mistakes, "of", len(results))
    print("accuracy percentage:", 100 - (mistakes * 100 / len(results)), "%")
    print("----------------------------------------------------------------------------------")

    coord.request_stop()
    coord.join(threads)
