#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri//mnist_mlp.py
# date: 2018-March-1         
#-----------------------------------------------------------------------------#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

#print_tensors_in_checkpoint_file('./log/model-2', tensor_name='', all_tensors='')

reader = tf.train.NewCheckpointReader('./log/model-2')
conv1_weights = reader.get_tensor('conv1/weights')
conv1_biases = reader.get_tensor('conv1/biasses')
conv2_weights = reader.get_tensor('conv2/weights')
conv2_biases = reader.get_tensor('conv2/biasses')
fc1_weights = reader.get_tensor('fc1/weights')
fc1_biases = reader.get_tensor('fc1/biasses')
fc2_weights = reader.get_tensor('fc2/weights')
fc2_biases = reader.get_tensor('fc2/biasses')


#plt.figure(1)
#for i in range(4):
#    plt.subplot(1,4,i+1)
#    plt.imshow(w1[:,:,0,i], interpolation='none', cmap='gray')


np.savez('model_parameters.npz',
         conv1_weights= conv1_weights,
         conv1_biases= conv1_biases,
         conv2_weights= conv2_weights,
         conv2_biases= conv2_biases,
         fc1_weights= fc1_weights,
         fc1_biases= fc1_biases,
         fc2_weights= fc2_weights,
         fc2_biases= fc2_biases,
         )





