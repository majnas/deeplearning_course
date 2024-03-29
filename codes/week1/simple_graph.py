#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/
# date: 2018-Feb-11            
#-----------------------------------------------------------------------------#

"""
    basic graph
"""

import numpy as np
import tensorflow as tf
    
# placeholders
X = tf.placeholder(tf.float32, shape=[10, 6])

# variables
W = tf.Variable(tf.random_normal([2, 10]), tf.float32)
b = tf.Variable(tf.random_normal([2, 6]), tf.float32)

# operations
WX = tf.matmul(W, X)
WXb = tf.add(WX, b)
relu = tf.nn.relu(WXb)

# create Session
with tf.Session() as sess:
        
    # init variables
    sess.run(tf.global_variables_initializer())
    
    # feed placeholders
    feed_dict = {X: np.random.randn(10,6)}
    relu_val = sess.run(relu, feed_dict=feed_dict)
    
