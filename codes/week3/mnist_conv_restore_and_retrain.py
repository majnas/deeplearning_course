#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/
# date: 2018-March-1         
#-----------------------------------------------------------------------------#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./MNIST', one_hot=True)
n_train = mnist.train.num_examples

log_dir='./log/model'


tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    loader = tf.train.import_meta_graph('./log/model-2.meta')
    loader.restore(sess, './log/model-2')    

    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    fc1_relu = graph.get_tensor_by_name("fc1/relu:0")
    
    tshape = fc1_relu.get_shape().as_list()
    n_classes = 10
    
    # add new fc2
    # Second Fully Connected Layer
    # (None, 120) --> (None, 10)
    with tf.name_scope('fc_new'):
        weights = tf.Variable(tf.truncated_normal([tshape[1], n_classes], mean=0.0, stddev=0.1), tf.float32, name='weights')
        biasses = tf.Variable(tf.constant(0.1, shape=[n_classes]), tf.float32, name='biasses')
        fc = tf.matmul(fc1_relu, weights)
        fc = tf.nn.bias_add(fc, biasses, name='fc')
        logits = tf.nn.softmax(fc)
        
    
    # loss
    loss_new = tf.reduce_mean(-Y * tf.log(logits), name='loss_new')
    
    # optimizer
    var_list = [v.name for v in tf.trainable_variables()]
    train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_new')
    #print(train_var)
      
#    train_var = [weights, biasses]
    opt_new = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_new, var_list=train_var)
    #
    # accuracy
    accuracy = tf.equal(tf.argmax(logits, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy, name='accuracy')
    tf.summary.scalar('accuracy', accuracy)
    
    # training variables
    n_epochs = 3
    batch_size = 128
    n_itrs = n_train // batch_size
    display_step = 10
        
    # initialize variables
    #sess.run(weights.initializer)
    #sess.run(biasses.initializer)
    un_init = [tf.is_variable_initialized(v) for v in tf.all_variables()]
    un_init = sess.run(un_init)
    un_init_vars = [v for (v, f) in zip(tf.all_variables(), un_init) if not f]
    init = tf.initialize_variables(var_list= un_init_vars)
    sess.run(init)

    
    saver = tf.train.Saver(max_to_keep=100)
    
    for epoch in range(n_epochs):
        for itr in range(n_itrs):
            
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            feed_dict = {X: batch_x, Y: batch_y}
            sess.run(opt_new, feed_dict=feed_dict)
            
            if ((itr+1) % display_step == 0):
                # evaluate for minibatch
                feed_dict = {X: batch_x, Y: batch_y}
                loss_val, acc_val= sess.run([loss_new, accuracy], feed_dict=feed_dict)
                
                print('epoch= ', epoch, 'minibatch loss= ', loss_val, 'minibatch acc= ', acc_val)
        
        # evaluate i4n each epoch
        feed_dict = {X: mnist.test.images,
                     Y: mnist.test.labels,}
        loss_val, acc_val = sess.run([loss_new, accuracy], feed_dict=feed_dict)
        
        print('epoch= ', epoch, 'epoch loss= ', loss_val, 'epoch acc= ', acc_val)
        
        
        saver.save(sess, log_dir, global_step=epoch)    
    
    
    
    sess.close()






