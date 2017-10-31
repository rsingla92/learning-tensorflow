# tf.estimator API
# simplifies the mechanics of machine learning such as running training
# loops, running evaluation loops, and managing data sets.
# It also defines many common models.

import tensorflow as tf

# NumPy is used to load, manipulate and preprocess data
import numpy as np

# A list of features. This only has one numeric feature.
# Other types of columns are more complicated and useful.

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# Estimator is the front end to invoke training/fitting and evaluation/inference
# Predefined types like linear regression, linear classification, and other NN
# classifiers and regressors.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TF has many helper methods to read and set up data sets.
# One data set here is for training, the other for evaluation.
# Need to guess how many epochs are desired and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# Invoke 1000 training steps by invoking the train method with the input training
# data set.
estimator.train(input_fn=input_fn, steps=1000)

# Evaluate the model in training and test
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("Training metrics: %r"% train_metrics)
print("Evaluation metrics: %r"% eval_metrics)
