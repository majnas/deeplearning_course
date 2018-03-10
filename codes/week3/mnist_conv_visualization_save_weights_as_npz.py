#-----------------------------------------------------------------------------#
# coder: Majid Nasiri
# github: https://github.com/m-nasiri/
# date: 2018-March-1         
#-----------------------------------------------------------------------------#



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


log_dir = './log/model-0'
loader = tf.train.NewCheckpointReader(log_dir)


conv1_w = loader.get_tensor('conv1/weights')
conv1_b = loader.get_tensor('conv1/biases')
conv2_w = loader.get_tensor('conv2/weights')
conv2_b = loader.get_tensor('conv2/biases')
fc1_w = loader.get_tensor('fc1/weights')
fc1_b = loader.get_tensor('fc1/biases')
fc2_w = loader.get_tensor('fc2/weights')
fc2_b = loader.get_tensor('fc2/biases')


plt.figure(1)
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(conv1_w[:,:,0,i], 
               interpolation='none',
               cmap='gray')


np.savez('model_param.npz',
         conv1_w = conv1_w,
         conv1_b = conv1_b,
         conv2_w = conv2_w,
         conv2_b = conv2_b,
         fc1_w = fc1_w,
         fc1_b = fc1_b,
         fc2_w = fc2_w,
         fc2_b = fc2_b,
         )



















