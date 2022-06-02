# Avoid console warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Node creation
node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)

tf_session = tf.Session() # we need to call tf_session()/sess() every time we execute a function

print(tf_session.run([node1, node2])) # print(node1, node2) only prints info of the nodes/tensors

node3 = tf.add(node1, node2)

# Computational graph
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
sess.run(adder_node, {a: 2, b: 6}) # prints 8
sess.run(adder_node, {a: [1, 2, 3], b: [4, 5, 6]}) # prints [5, 7, 9]

# Linear model
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer() # initialize all the tf.Variable()
sess.run(init)

# Use sess.run(linear_model, {x: <any value>}) to test

# A loss function measures how far apart the current model is from the
# provided data. We'll use a standard loss model for linear regression, (yhat - y)^2
# which sums the squares of the deltas between the current model and the
# provided data. linear_model - y creates a vector where each element is
# the corresponding example's error delta. We call tf.square to square
# that error. Then, we sum all the squared errors to create a single scalar
# that abstracts the error of all examples using tf.reduce_sum:
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y) # (yhat - y)^2
loss = tf.reduce_sum(squared_deltas) # run using sess.run(loss, {x: <any value>, b: <any value>})

# Reassign values using
# foo = tf.assign(W, [any value])
# don't forget to call sess.run(foo) every time you declare or do initialization

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [-1, -2, -3, -5]})

# Test using sess.run([W, b])
