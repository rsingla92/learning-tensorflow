
"""
    Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - builds the model as far as required for running the network
    forward to make predictions.

2. loss() - adds to the inference model the layers required to generate loss

3. training() - adds to the loss model the ops requires to generate and apply
    gradients.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

# Classes in MNIST
NUM_CLASSES = 10

# MNIST image sizes
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def inference(images, hidden1_units, hidden2_units):
    """
    Build the MNIST model to where it may be used for inference.

    :param images: images placeholder from inputs()
    :param hidden1_units: size of the first hidden layer
    :param hidden2_units: size of the second hidden layer
    :return softmax_linear: output tensor with computed logits
    """

    # Each layer is prefixed with the scope and a name for that scope.
    # Within each layer, the weights and biases to be used are defined.

    # Commonly, weights are initialized with tf.truncated_normal
    # and given the shape of a 2D tensor with units in the layer FROM
    # which they connect and then the layer TO which weights connect.

    # Biases are initalized to zero, and have the number of units TO which they
    # connect.
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0/ math.sqrt(float(IMAGE_PIXELS))), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss(logits, labels):
    """
    Calculates the loss from the logits and labels.

    :param logits: logits tensor, float - [batch_size, NUM_CLASSES]
    :param labels: labels tensor, int32 - [batch_size]
    :return: loss tensor, float.
    """

    # Labels are first converted to 64-bit integers.
    labels = tf.to_int64(labels)

    # Automatically produce 1-hot labels from the placeholder and compare the
    # logits (given from inference()) with 1-hot labels.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
    """
    Sets up the training ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.

    Op returned is what should be passed to sess.run() to train the model.

    :param loss:  loss tensor, resulting from loss() call
    :param learning_rate: learning rate for gradient descent to use
    :return: the Op for training
    """

    # Minimize loss via Gradient Descent, using the given learning rate.

    tf.summary.scalar('loss', loss) # snapshot loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Apply the gradient that minimize the loss as a single training step
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label

    :param logits: logits tensor, float - [batch_size, NUM_CLASSES]
    :param labels: tensor, int32 - [batch_size] with values in the range [0, NUM_CLASSES]
    :return: scalar int32 tensor with the number of examples (out of batch_size) that were
            predicted correctly.
    """

    # For classification, we will use the in_top_k operation.
    # This returns a boolean tensor with shape [batch_size] that is true
    # for examples where the label is in the top k of all logits for that
    # example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

