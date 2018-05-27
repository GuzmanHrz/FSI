import gzip
import cv2
import _pickle
import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set


# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt


#lt.show()  # Let's see a sample
#print(train_x[57])
#print (train_y[57])

# TODO: the neural net!!
y_data = one_hot(train_y , 10)
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20
valid_x, valid_y = valid_set


error = 0
perror = 0
epoch = 0
epochs = []
errors = []
while abs(perror - error) >= perror * 0.0001:
    for jj in range(len(train_x) // batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    perror = error
    error= sess.run(loss, feed_dict={x: valid_x, y_: one_hot(valid_y,10)})
    errors.append(error)
    epochs.append(epoch)

    print("Epoch #:", epoch, "Error: ", error)
    epoch += 1

plt.plot(epochs, errors)
plt.show()


print ("----------------------")
print ("   Test               ")
print ("----------------------")

test_x, test_y = test_set

result = sess.run(y, feed_dict={x:  test_x})
mistakes = 0
for b, r in zip(test_y , result):
    if b != np.argmax(r):
        mistakes += 1
        print ( b, "-->", np.argmax(r) )
print ("accuracy percentage:", 100 - (mistakes * 100 / len(test_y)), "%")
print ("----------------------------------------------------------------------------------")
