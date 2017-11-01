"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

from six.moves import xrange
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

# Basic model parameters as external flags
FLAGS = None


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
     These placeholders are used as inputs by the rest of the model building
     code and will be fed from the downloaded data in the .run() loop, below.
     Args:
       batch_size: The batch size will be baked into both placeholders.
     Returns:
       images_placeholder: Images placeholder.
       labels_placeholder: Labels placeholder.
     """
    # Note that the shapes of the placeholders match the the shapes of the full
    # image and label tesnsors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.

    # Two placeholder ops that define input shapes (and batch_size).
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    """
    Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed>,
        ...
    }

    :param data_set:  the set of images and labels from input_data.read_data_sets()
    :param images_pl: the images placeholder, from placeholder_inputs()
    :param labels_pl: labels placeholder, from placeholder_inputs().
    :return:  feed_dict: dictionary mapping placeholders to values
    """

    # Data is queried for next batch_size st of images and labels
    # Tensors matching the placeholders are filled, and a dict object
    # is generated.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """

    # Run a full single epoch of evaluation
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)

        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    precision = float(true_count) / num_examples
    print('Num examples: %d\nNum correct: %d\nPrecision @1: %0.04f' % (num_examples, true_count, precision))


def run_training():
    """
    Train MNIST for a number of steps.
        """

    # Ensures the correct data has been downloaded and unpacks it into a dict of
    # DataSet instances.
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    # Tell TF that the model will be built into the default Graph.
    # 'with' command indicates all of the ops are associated with the specified
    # instance - this being the default global tf.Graph instance
    # A tf.Graph is a collection of ops that may be executed together as a group.
    with tf.Graph().as_default():
        # Generate placeholders
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a graph that computes predictions from the inference model.
        # Inference function builds the graph as far as needed to return the tensor
        #   containing output predictions.
        # Takes images placeholder in and builds on top a pair of fully connected layers.
        #   using ReLU activation. It then has a ten node linear layer with outputs.
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

        # Add the ops for loss calculation
        loss = mnist.loss(logits, labels_placeholder)

        # Add ops that calculate and apply gradients
        train_op = mnist.training(loss, FLAGS.learning_rate)

        # Add op to compare logits to labels during evaluation
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Summary tensor based on collection of summaries
        summary = tf.summary.merge_all()

        # Add the variable initalizer
        init = tf.global_variables_initializer()

        # Create a saver
        saver = tf.train.Saver()

        # Create a session for running ops
        # Alternatively, could do 'with tf.Session() as sess:'
        sess = tf.Session()

        # Instantiate SummaryWriter for output
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        ### Built everything ! ###

        # Now run and train.
        # run() will complete the subset of the graph as corresponding to the
        #   ops described above. Thus, only init() is given.
        sess.run(init)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with actual set of images
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)

            # Run a step.
            # What is returned is the activations from the training_op
            # and the loss operation.
            # If you want to insepct the values of ops or variables, include
            # them in the list passed to sess.run()

            # Each tensor in the list of values corresponds to a numpy array in the returned tuple.
            # This is filled with the value of that tensor during this step of training.
            # Since train_op is an Operation with no output value, it can be discarded.
            # BUT...if loss becomes NaN, the model has likely diverged during training.


            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Let's log some stuff so we know we're doing ok.
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

                #Update events file
                # This can be used by TensorBoard  to display the summaries.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)

            print('Training Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.train)
            # Evaluate against the validation set.
            print('Validation Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.validation)
            # Evaluate against the test set.
            print('Test Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.test)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/fully_connected_feed'),
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


















