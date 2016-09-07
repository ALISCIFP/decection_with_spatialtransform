# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from PIL import Image     

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu,method = "xavier"):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim],method = method,name = layer_name)
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    if act is None:
      activations = preactivate
    else:
      activations = act(preactivate, 'activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations

def save_as_image( img ,iter_i,epoch_i,type):
    "save an array as image"
    img1 = img[0,:,:,0]
    img1 = img1.reshape((40,40))
    img2 = img[0,:,:,1]
    img2 = img2.reshape((40,40))

    
    contenated_image = np.concatenate((img1,img2), axis=0)

    contenated_image = contenated_image * 256
    contenated_image = Image.fromarray(contenated_image.astype(np.uint8),'L')
    contenated_image.save('try/'+'img'+' '+str(epoch_i) +'_' + str(iter_i) + type+ '.bmp')
    return 
# %% Load data
mnist_cluttered = np.load('./data/mnist_sequence1_sample_5distortions5x5_2channel.npz')

X_train = mnist_cluttered['X_train']
y_train = mnist_cluttered['y_train']
X_valid = mnist_cluttered['X_valid']
y_valid = mnist_cluttered['y_valid']
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']

# % turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=19)
Y_valid = dense_to_one_hot(y_valid, n_classes=19)
Y_test = dense_to_one_hot(y_test, n_classes=19)

# %% Graph representation of our network

# %% Placeholders for 40x40 resolution
x = tf.placeholder(tf.float32, [None,3200])
y = tf.placeholder(tf.float32, [None, 19])

# %% Since x is currently [batch, height*width], we need to reshape to a
# 4-D tensor to use it in a convolutional graph.  If one component of
# `shape` is the special value -1, the size of that dimension is
# computed so that the total size remains constant.  Since we haven't
# defined the batch dimension's shape yet, we use -1 to denote this
# dimension should not change size.
x_tensor = tf.reshape(x, [-1, 80, 40, 1])
# test if the dimension layout is correct
# XX = X_train.reshape((-1,40,40,2))
# img = XX[10,:,:,0]
# img = img * 256
# img = Image.fromarray(img.astype(np.uint8),'L')
# img.show()
 

# %% We'll setup the two-layer localisation network to figure out the
# %% parameters for an affine transformation of the input
# %% Create variables for fully connected layer

# %% We can add dropout for regularizing and to reduce overfitting like so:
keep_prob = tf.placeholder(tf.float32)
# %% Second layer

ful_length = 20

h_fc_loc11 = nn_layer(x, 3200, ful_length, 'fc_loc11',act = tf.nn.tanh,method = 'zeros')
h_fc_loc11_drop = tf.nn.dropout(h_fc_loc11, keep_prob)

with tf.name_scope('fc_loc12'):
    with tf.name_scope('weights'):
        W_fc_loc12 = weight_variable([ful_length, 6],method = 'zeros')
        variable_summaries(W_fc_loc12, 'fc_loc12' + '/weights')
# Use identity transformation as starting point
    with tf.name_scope('biases'):
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        b_fc_loc12 = tf.Variable(initial_value=initial, name='b_fc_loc12')
        variable_summaries(b_fc_loc12, 'fc_loc12' + '/biases')

    with tf.name_scope('Wx_plus_b'):
      fc_loc12_preactivate = tf.matmul(h_fc_loc11_drop, W_fc_loc12) + b_fc_loc12
      tf.histogram_summary('fc_loc12' + '/pre_activations',fc_loc12_preactivate)     
    h_fc_loc12 = tf.nn.tanh(fc_loc12_preactivate, 'activation')
    tf.histogram_summary('fc_loc12' + '/activations', h_fc_loc12)  
    

with tf.name_scope('fc_loc22'):
    with tf.name_scope('weights'):
        W_fc_loc22 = weight_variable([ful_length, 6],method = 'zeros')
        variable_summaries(W_fc_loc22, 'fc_loc22' + '/weights')

# Use identity transformation as starting point
    with tf.name_scope('biases'):
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        b_fc_loc22 = tf.Variable(initial_value=initial, name='b_fc_loc22')
        variable_summaries(b_fc_loc22, 'fc_loc22' + '/biases')

    with tf.name_scope('Wx_plus_b'):
      fc_loc22_preactivate = tf.matmul(h_fc_loc11_drop, W_fc_loc22) + b_fc_loc22
      tf.histogram_summary('fc_loc22' + '/pre_activations',fc_loc22_preactivate)     
    h_fc_loc22 = tf.nn.tanh(fc_loc22_preactivate, 'activation')
    tf.histogram_summary('fc_loc22' + '/activations', h_fc_loc22)  
    
# %% We'll create a spatial transformer module to identify discriminative
# %% patches
out_size = (40, 40,1)
h_trans_0 = transformer(x_tensor, h_fc_loc12, out_size)
h_trans_1 = transformer(x_tensor, h_fc_loc22, out_size)
h_trans = tf.concat(3, [h_trans_0, h_trans_1])
show_tranformed = tf.reshape(h_trans,[-1,40,40,1])
tf.image_summary("transformed images", show_tranformed, max_images=4)
# %% We'll setup the first convolutional layer
# Weight matrix is [height x width x input_channels x output_channels]

# # %% We'll now reshape so we can connect to a fully-connected layer:
h_conv2_flat = tf.reshape(h_trans, [-1, 3200])
h_fc1 = nn_layer(h_conv2_flat, 3200, 512, 'fcnn_1')
h_fc2 = nn_layer(h_fc1, 512, 512, 'fcnn_2')

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


# %% And finally our softmax layer:
y_logits = nn_layer(h_fc2_drop, 512, 19, 'fcnn_3',act=None)
 
# %% Define loss/eval/training functions
with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
    tf.scalar_summary('cross entropy', cross_entropy)

with tf.name_scope('train'):
    opt = tf.train.AdamOptimizer(learning_rate=0.0019)
    optimizer = opt.minimize(cross_entropy)
grads = opt.compute_gradients(cross_entropy, [b_fc_loc12,b_fc_loc22])

# %% Monitor accuracy
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    tf.scalar_summary('accuracy', accuracy)

# %% We now create a new session to actually perform the initialization the
# variables:
merged = tf.merge_all_summaries()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
train_writer = tf.train.SummaryWriter('tensorboard_out' + '/train',
                                      sess.graph)
test_writer = tf.train.SummaryWriter('tensorboard_out'+'/test',sess.graph)


print sess.run(tf.initialize_all_variables())

# %% We'll now train in minibatches and report accuracy, loss:
iter_per_epoch = 8000
n_epochs = 500
train_size = 800000

indices = np.linspace(0, 800000 - 1, iter_per_epoch)
indices = indices.astype('int')

for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

        if iter_i % 1000 == 0:
            summary,loss = sess.run([merged,cross_entropy],
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            train_writer.add_summary(summary, epoch_i*iter_per_epoch + iter_i)
            print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

            img = sess.run(x_tensor,
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            img = np.concatenate((img[0:1,0:40,:,:],img[0:1,40:80,:,:]), axis=3)
            save_as_image( img ,iter_i,epoch_i,'origin')    


            img = sess.run(h_trans,
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            save_as_image( img ,iter_i,epoch_i,'transformed')   

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.8})
            #train_writer.add_summary(summary, epoch_i*iter_per_epoch + iter_i)
    acc,summary = sess.run([accuracy,merged],
          feed_dict={
          x: X_valid[0:1000],
          y: Y_valid[0:1000],
          keep_prob: 1.0
                      })
    test_writer.add_summary(summary, epoch_i*iter_per_epoch + iter_i)
    print('Accuracy (%d): ' % epoch_i + str(acc))
    
    # theta = sess.run(h_fc_loc2, feed_dict={
    #        x: batch_xs, keep_prob: 1.0})
    # print(theta[0])