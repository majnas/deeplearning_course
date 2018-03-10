#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri//mnist_mlp.py
# date: 2018-March-1         
#-----------------------------------------------------------------------------#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('./MNIST', one_hot=True)


tf.reset_default_graph()
sess = tf.Session()
loader = tf.train.import_meta_graph('./log/model-2.meta')
loader.restore(sess, './log/model-2')
graph = tf.get_default_graph()

X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
loss = graph.get_tensor_by_name("loss:0")
accuracy = graph.get_tensor_by_name("accuracy:0")


loss_val, accuracy_val = sess.run([loss, accuracy], feed_dict={X: mnist.test.images,
                                                               Y: mnist.test.labels,
                                                               keep_prob: 1.0})

print('loss', loss_val, accuracy_val)

sess.close()






