from model import SRCNN
from utils import input_setup

import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("num_iter", 1000, "Number of iteration [1000]")
flags.DEFINE_integer("epoch", 10, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", .0001, "The learning rate [.0001]")
flags.DEFINE_float("momentum", .9, "The size of momentum [.9]")
flags.DEFINE_float("weight_decay", 0, "The size of weight decay [0]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [3]")
flags.DEFINE_float("stddev", .001, "Stddev of initializing weight parameters [.001]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  with tf.Session() as sess:
    srcnn = SRCNN(sess, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)

    if FLAGS.is_train:
      srcnn.train(FLAGS)
    else:
      srcnn.load(FLAGS.checkpoint_dir)

if __name__ == '__main__':
  tf.app.run()