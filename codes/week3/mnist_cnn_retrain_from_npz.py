#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/tensorflow/blob/master/basic/mnist_cnn_dropout_summary.py
# date: 2018-Feb-11            
#-----------------------------------------------------------------------------#

"""
    basic convolutional neural network to classify mnist dataset + droupout + 
    using tensorboard
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('./MNIST', one_hot=True)

n_train = mnist.train.num_examples
n_test = mnist.train.num_examples

log_dir='./log2/model'


params = np.load('model_parameters.npz')


# network parameters
nF1 = 4
nF2 = 8
nFc1 = 120
n_classes = 10


tf.reset_default_graph()

# place holders
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 10], name='Y')
X_ = tf.reshape(X, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# First Convolutional Layer
# (None, 28, 28, 1) --> (3, 3, 1, 8) --> (None, 28, 28, 8) --> (None, 14, 14, 8)
with tf.name_scope('conv1'):
    weights = tf.Variable(params['conv1_weights'], tf.float32, name='weights')
    biasses = tf.Variable(params['conv1_biases'], tf.float32, name='biasses')
    conv = tf.nn.conv2d(X_, weights, strides=[1,1,1,1], padding='SAME')
    conv = tf.nn.bias_add(conv, biasses, name='conv')
    relu = tf.nn.relu(conv, name='relu')
    pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    tf.summary.histogram('actvation', conv)

# Second Convolutional Layer
# (None, 14, 14, 8) --> (3, 3, 8, 12) --> (None, 14, 14, 12) --> (None, 7, 7, 12)
with tf.name_scope('conv2'):
    weights = tf.Variable(params['conv2_weights'], tf.float32, name='weights')
    biasses = tf.Variable(params['conv2_biases'], tf.float32, name='biasses')
    conv = tf.nn.conv2d(pool, weights, strides=[1,1,1,1], padding='SAME')
    conv = tf.nn.bias_add(conv, biasses, name='conv')
    relu = tf.nn.relu(conv, name='relu')
    pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    tf.summary.histogram('actvation', conv)

    
pool_shape = pool.get_shape().as_list()
n_flat = pool_shape[1] * pool_shape[2] * pool_shape[3]
# (None, 7, 7, 12) --> (None, 7*7*12)
pool_flat = tf.reshape(pool, shape=[-1, n_flat])

# First Fully Connected Layer
# (None, 7*7*12) --> (None, 120)
with tf.name_scope('fc1'):
    weights = tf.Variable(params['fc1_weights'], tf.float32, name='weights')
    biasses = tf.Variable(params['fc1_biases'], tf.float32, name='biasses')
    fc = tf.matmul(pool_flat, weights)
    fc = tf.nn.bias_add(fc, biasses, name='fc')
    relu = tf.nn.relu(fc, name='relu')
    fc = tf.nn.dropout(relu, keep_prob=keep_prob)
    tf.summary.histogram('actvation', fc)

    
# Second Fully Connected Layer
# (None, 120) --> (None, 10)
with tf.name_scope('fc2'):
    weights = tf.Variable(params['fc2_weights'], tf.float32, name='weights')
    biasses = tf.Variable(params['fc2_biases'], tf.float32, name='biasses')
    fc = tf.matmul(fc, weights)
    fc = tf.nn.bias_add(fc, biasses, name='fc')
    logits = tf.nn.softmax(fc)
    tf.summary.histogram('actvation', fc)
    tf.summary.histogram('logits', logits)

# loss
loss = -tf.reduce_mean(Y * tf.log(logits))
tf.summary.scalar('loss', loss)

# optimizer
opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# accuracy
accuracy = tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1))
accuracy = tf.cast(accuracy, tf.float32)
accuracy = tf.reduce_mean(accuracy)
tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()

# training variables
n_epochs = 1
batch_size = 128
n_itrs = n_train // batch_size
display_step = 10

with tf.Session() as sess:
    
    # initialize variables
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter(log_dir)
    writer.add_graph(sess.graph)
    
    saver= tf.train.Saver(max_to_keep=10)
    
    for epoch in range(n_epochs):
        for itr in range(n_itrs):
            
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            feed_dict = {X: batch_x, Y: batch_y, keep_prob: 0.8}
            sess.run(opt, feed_dict=feed_dict)
            
            if ((itr+1) % display_step == 0):
                # evaluate for minibatch
                feed_dict = {X: batch_x, Y: batch_y, keep_prob: 1.0}
                loss_val, acc_val, summary = sess.run([loss, accuracy, merged_summary], feed_dict=feed_dict)
                
                print('epoch= ', epoch, 'minibatch loss= ', loss_val, 'minibatch acc= ', acc_val)
                writer.add_summary(summary, epoch * n_itrs + itr)
        
        # evaluate in each epoch
        feed_dict = {X: mnist.test.images,
                     Y: mnist.test.labels,
                     keep_prob: 1.0}
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict=feed_dict)
        
        saver.save(sess, log_dir, global_step=epoch)
        
        print('epoch= ', epoch, 'epoch loss= ', loss_val, 'epoch acc= ', acc_val)



