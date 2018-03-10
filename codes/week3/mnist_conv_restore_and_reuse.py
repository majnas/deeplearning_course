#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/
# date: 2018-March-1         
#-----------------------------------------------------------------------------#


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST', one_hot=True)
n_train = mnist.train.num_examples



tf.reset_default_graph()

sess = tf.Session()
loader = tf.train.import_meta_graph('./log/model-0.meta')
loader.restore(sess, './log/model-0')

graph = tf.get_default_graph()

X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
loss = graph.get_tensor_by_name("loss:0")


batch_size = 128
#train_x, train_y = mnist.train.next_batch(batch_size)
            
loss_val = sess.run(loss, feed_dict={X: mnist.test.images,
                                     Y: mnist.test.labels,
                                     keep_prob: 1.0})

print(loss_val)


sess.close()




























