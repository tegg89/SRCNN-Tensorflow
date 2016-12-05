from utils import read_data, input_setup, save_images

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

def conv2d(input_, output_dim, 
           k_h, k_w, d_h=1, d_w=1, stddev=1e-3, 
           name='conv2d', reuse=False):
  if reuse:
    with tf.variable_scope(name):
      w = tf.Variable(tf.random_normal([k_h, k_w, input_.get_shape()[-1], output_dim]), name='w')
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

      biases = tf.Variable(tf.zeros([output_dim]), name='biases')
      conv = tf.nn.bias_add(conv, biases)
      return conv
  else:
    with tf.variable_scope(name):
      w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], 
                          initializer=tf.truncated_normal_initializer(stddev=stddev))
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

      biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.))
      conv = tf.nn.bias_add(conv, biases)
      return conv
    

class SRCNN(object):

  def __init__(self, sess, image_size=33, label_size=21, c_dim=1, checkpoint_dir=None, sample_dir=None):

    self.sess = sess
    self.image_size = image_size
    self.label_size = label_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], 
                                 name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], 
                                 name='labels')
    
    self.conv1 = tf.nn.relu(conv2d(self.images, 64, 9, 9, name='conv1')) # [None, 25, 25, 64]
    self.conv2 = tf.nn.relu(conv2d(self.conv1, 32, 1, 1, name='conv2')) # [None, 25, 25, 32]
    self.conv3 = tf.nn.relu(conv2d(self.conv2, 1, 5, 5, name='conv3')) # [None, 21, 21, 1]

    self.sampler = self.sampler()

    self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.labels, self.conv3))))

    t_vars = tf.trainable_variables()

    self.var_list = [var for var in t_vars if 'conv' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    input_setup(self.sess)

    data_dir = os.path.join(os.sep, os.getcwd(), "checkpoint/train_.h5")
    train_data, train_label = read_data(data_dir)

    train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss, var_list=self.var_list)
    tf.initialize_all_variables().run()

    counter = 1
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for ep in xrange(config.epoch):
      # Batch files...
      # 
      for idx in xrange(0, config.num_iter):

        err = self.loss.eval({self.images: train_data, self.labels: train_label})

        counter += 1
        print('Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]' \
          % (ep, idx, time.time()-start_time, err))

        if np.mod(counter, 100) == 1:
          samples, loss = self.sess.run([self.sampler, self.loss], feed_dict={self.images: train_images, self.labels: train_label})
          save_images(samples, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, ep, idx))
          print("[Sample] loss: %.8f" % loss)

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def sampler(self):
    tf.get_variable_scope().reuse_variables()

    l1 = tf.nn.relu(conv2d(self.images, 64, 9, 9, name='conv1', reuse=True))
    l2 = tf.nn.relu(conv2d(l1, 32, 1, 1, name='conv2', reuse=True))
    l3 = tf.nn.relu(conv2d(l2, 1, 5, 5, name='conv3', reuse=True))

    return l3
  
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
