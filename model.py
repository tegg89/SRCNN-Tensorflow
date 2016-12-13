from utils import (
  read_data, 
  input_setup, 
  load_sample,  
  psnr
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

def conv2d(input_, output_dim, 
           k_h, k_w, d_h=1, d_w=1, stddev=1e-3, 
           name='conv2d', reuse=False):
  if reuse:
    with tf.variable_scope(name, reuse=reuse) as scope:
      tf.get_variable_scope().reuse_variables()
      # Open a reusing scope.
      assert tf.get_variable_scope().reuse == True

      w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], 
                          initializer=tf.random_normal_initializer(stddev=stddev))
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

      biases = tf.get_variable('biases', [output_dim], 
                               initializer=tf.constant_initializer(0.))
      conv = tf.nn.bias_add(conv, biases)
      return conv
  else:
    with tf.variable_scope(name):
      # At start, the scope is not reusing.
      assert tf.get_variable_scope().reuse == False

      w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], 
                          initializer=tf.random_normal_initializer(stddev=stddev))
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

      biases = tf.get_variable('biases', [output_dim],
                               initializer=tf.constant_initializer(0.))
      conv = tf.nn.bias_add(conv, biases)
      return conv    

class SRCNN(object):

  def __init__(self, 
               sess, 
               image_size=33,
               label_size=21, 
               batch_size=64,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], 
                                 name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], 
                                 name='labels')
    
    self.model = self.cnn(reuse=False)

    self.sampler = self.cnn(reuse=True)
    
    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.model))

    t_vars = tf.trainable_variables()

    self.var_list = [var for var in t_vars if 'conv' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    """Train SRCNN"""
    input_setup(self.sess) # To create input h5 file

    sample_data, sample_label = load_sample(self.sess, 0)

    data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    train_data, train_label = read_data(data_dir)

    # Stochastic gradient descent with the standard backpropagation
    train_op = tf.train.GradientDescentOptimizer(config.learning_rate) \
                       .minimize(self.loss, var_list=self.var_list)
    tf.initialize_all_variables().run()

    counter = 1
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for ep in xrange(config.epoch):
      # Run by batch images
      batch_idxs = len(train_data) // config.batch_size

      for idx in xrange(0, batch_idxs):
        batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
        batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

        err = self.loss.eval({self.images: batch_images, self.labels: batch_labels})
        result = self.model.eval({self.images: batch_images, self.labels: batch_labels})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % (ep, idx, batch_idxs, counter, time.time()-start_time, err))

        if np.mod(counter, 100) == 1:
          samples, loss = self.sess.run(
              [self.sampler, self.loss], 
              feed_dict={self.images: sample_data, self.labels: sample_label}
          )

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def cnn(self, reuse):
    conv1 = tf.nn.relu(conv2d(self.images, 64, 9, 9, stddev=1e-3, name='conv1', reuse=reuse)) # [None, 25, 25, 64]
    conv2 = tf.nn.relu(conv2d(conv1, 32, 1, 1, stddev=1e-3, name='conv2', reuse=reuse)) # [None, 25, 25, 32]
    conv3 = (conv2d(conv2, 1, 5, 5, stddev=1e-4, name='conv3', reuse=reuse)) # [None, 21, 21, 1]

    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
