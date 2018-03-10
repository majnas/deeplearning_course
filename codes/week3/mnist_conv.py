#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/
# date: 2018-March-1         
#-----------------------------------------------------------------------------#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets('./MNIST', one_hot=True)
n_train = mnist.train.num_examples

log_dir = './log/model'

tf.reset_default_graph()

#placeholders
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
X_ = tf.reshape(X, shape=[-1,28,28,1])
Y = tf.placeholder(tf.float32, shape=[None, 10], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# first convolutional layer
with tf.name_scope('conv1'):
    weights=tf.Variable(tf.truncated_normal(shape=[3,3,1,8], 
                                            mean=0, 
                                            stddev=0.1), 
                                            dtype=tf.float32,
                                            name='weights')
    biases=tf.Variable(tf.constant(0.1, shape=[8]), 
                                   dtype=tf.float32,
                                   name='biases') 
    conv=tf.nn.conv2d(input=X_, 
                      filter=weights, 
                      strides=[1,1,1,1], 
                      padding='SAME')
    conv = tf.nn.bias_add(conv, biases)
    relu = tf.nn.relu(conv, name='relu')
    pool = tf.nn.max_pool(value=relu, 
                          ksize=[1,2,2,1], 
                          strides=[1,2,2,1], 
                          padding='SAME',
                          name='pool')


print('conv shape=', conv.get_shape())
print('pool shape=', pool.get_shape())

# second convolutional layer
with tf.name_scope('conv2'):
    weights=tf.Variable(tf.truncated_normal(shape=[3,3,8,12], 
                                            mean=0, 
                                            stddev=0.1), 
                                            dtype=tf.float32,
                                            name='weights')
    biases=tf.Variable(tf.constant(0.1, shape=[12]), 
                                   dtype=tf.float32,
                                   name='biases') 
    conv=tf.nn.conv2d(input=pool, 
                      filter=weights, 
                      strides=[1,1,1,1], 
                      padding='SAME')
    conv = tf.nn.bias_add(conv, biases)
    relu = tf.nn.relu(conv, name='relu')
    pool = tf.nn.max_pool(value=relu, 
                          ksize=[1,2,2,1], 
                          strides=[1,2,2,1], 
                          padding='SAME',
                          name='pool')


n_flat = 7*7*12
pool_flat = tf.reshape(pool, shape=[-1, n_flat])

# first fully-connected layer
with tf.name_scope('fc1'):
    weights=tf.Variable(tf.truncated_normal(shape=[n_flat,120], 
                                            mean=0, 
                                            stddev=0.1), 
                                            dtype=tf.float32,
                                            name='weights')
    biases=tf.Variable(tf.constant(0.1, shape=[120]), 
                                   dtype=tf.float32,
                                   name='biases') 
    
    fc = tf.matmul(pool_flat, weights)
    fc = tf.nn.bias_add(fc, biases)
    relu = tf.nn.relu(fc, name='relu')
    relu_drop = tf.nn.dropout(relu, keep_prob=keep_prob, name='relu_drop')
    

# second fully-connected layer
with tf.name_scope('fc2'):
    weights=tf.Variable(tf.truncated_normal(shape=[120,10], 
                                            mean=0, 
                                            stddev=0.1), 
                                            dtype=tf.float32,
                                            name='weights')
    biases=tf.Variable(tf.constant(0.1, shape=[10]), 
                                   dtype=tf.float32,
                                   name='biases') 
    
    fc = tf.matmul(relu_drop, weights)
    logits = tf.nn.bias_add(fc, biases, name='logits')
    pred = tf.nn.softmax(logits, name='pred')
    

# loss
loss = -tf.reduce_mean(Y * tf.log(pred+1e-8), name='loss')

# optimizer
opt = tf.train.AdamOptimizer(learning_rate=0.01)
opt = opt.minimize(loss, name='opt')



# accuracy
acc = tf.equal(tf.argmax(pred, axis=1),
               tf.argmax(Y, axis=1))

acc = tf.cast(acc, tf.float32)
acc = tf.reduce_mean(acc)





# training parameters
n_epochs = 1
batch_size = 128
n_itrs = n_train // batch_size
display_step =20

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(max_to_keep=10)
    
    for epoch in range(n_epochs):
        for itr in range(n_itrs):
            
            train_x, train_y = mnist.train.next_batch(batch_size)
            
            sess.run(opt, feed_dict={X: train_x,
                                     Y: train_y,
                                     keep_prob: 0.8})
            
            if (itr % display_step == 0):
                loss_val, acc_val = sess.run([loss, acc], feed_dict={X: train_x,
                                                     Y: train_y,
                                                     keep_prob: 1.0})
                print('itr=', itr, 'minibatch loss= ', loss_val, 'minibatch acc= ', acc_val)
                
                
        loss_val, acc_val = sess.run([loss, acc], feed_dict={X: mnist.test.images,
                                             Y: mnist.test.labels,
                                             keep_prob: 1.0})

                
        print('epoch=', epoch, 'loss= ', loss_val, 'acc= ', acc_val)
        
        saver.save(sess, log_dir, global_step=epoch)
    
    
    

print('conv shape=', conv.get_shape())
print('pool shape=', pool.get_shape())
print('pool_flat shape=', pool_flat.get_shape())
print('fc shape=', fc.get_shape())
print('pred shape=', pred.get_shape())

























