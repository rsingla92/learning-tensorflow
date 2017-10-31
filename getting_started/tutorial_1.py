# Canonical import statement for TF programs

import tensorflow as tf

# Step 1: Build the computational graph
# Step 2: Run the computational graph

# Each node takes zero or more tensors as input.

# There are constant types for nodes.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32, by default

# Print does not output values but describes nodes instead.
print(node1, node2)

# To get the output of 3.0 and 4.0, you need to run a session.
sess = tf.Session()
print(sess.run([node1, node2]))

# More complicated Tensor nodes are ones with operations.
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# Rather than create a graph with a known constant result, let's
# create something more interesting. Let us paramterize the
# the computational graph to accept external inputs.
# These are known as placeholders. It promises a value later.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # the operand + is a shortcut for tf.add(a,b)

# We can evaluate the graph with multiple inputs using the feed_dict
# arg in the run method
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# Add another operation for even more complexity!
add_and_triple = adder_node * 3.0
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# To make the model trainable, the graph needs to be modified so that
# new outputs are created with the same input. We use Variables
# to add trainable parameters to the graph.

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Must initialize the Variables! But how? Using the initializer.
# Init is a handle to the Tensorflow subgraph.
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# But how good is the model?! Evaluation is needed for sure.
# We need a placeholder to provide these values, and we need
# a loss function - a measure of how far apart the model is from
# the original data.

y = tf.placeholder(tf.float32)

# This is a vector where each element corresponds to
# an example's error's delta, and then squared and summed.
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# But we wanna find these parameters automatically...not guess!

# tf.train API
# Optimizers slowly change each variable to minimize the loss.
# The simplest one is gradient descent, modifying each variable
# according to the magnitude of the derivative of loss with respect
# to that variable.

# TF can automatically produce derivatives given a model description.

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data to use
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Training loop...
sess.run(init) # reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# What is the training accuracy?
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
