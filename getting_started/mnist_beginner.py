
# Download the MNIST data set automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# MNIST has training, test, and validation data.

# For now, let's ignore the spatial information about the images
# and just put them in a nice flat array (1x784).

# One-hot vectors is a vector which is 0 in most dimensions,
# and 1 in a single dimension. The nth digit is represented by
# a 1 in the nth dimension.

# Softmax regressions - useful when you want to assign probabilities to
# an object being one of several different things. The sum of softmax
# Step 1 - add up evidence of input being in a certain class
# Step 2 - convert the evidence into probabilities

# To sum up evidence, we do a weighted sum of the pixel intensities. The
# weight is negative if a pixel with high intensity is evidence against
# the class and positive if in favor.
# We also add some evidence called a bias for some level of input independence

# The probabilities themselves come from the following
#   softmax(x) = normalize(exp(x))

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# We're going to learn W and b, so it's not a big deal if they're 0 to start.
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

# Going to use cross-entropy loss.
# Compute the log of each element of y.
# Multiply each element of y_ by the log(y).
# Add the elements in the second dimension of y, specified by the reduction indices param
# Compute the mean of all the examples.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Learning rate of 0.5 for gradient descent.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Batch one hundred random training examples, as per stochastic training.
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


