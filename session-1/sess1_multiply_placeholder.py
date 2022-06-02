import tensorflow as tf 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

sess = tf.InteractiveSession()

a = tf.placeholder(tf.int32, name='a')
b = tf.placeholder(tf.int32, name='b')

multiply = tf.multiply(a, b)

print(multiply.eval(feed_dict={a: [2, 8], b: [6, 3]}))

