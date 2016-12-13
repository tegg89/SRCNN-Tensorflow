"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import scipy.misc
import os
import glob
import h5py
from PIL import Image  # for loading images as YCbCr format
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def read_data(path):
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, scale=3):
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)
  h, w = label_.shape

  input_ = scipy.misc.imresize(label_, size=(1./scale), interp='bicubic')
  input_ = scipy.misc.imresize(input_, (h, w), interp='bicubic')

  return input_, label_

def prepare_data(dataset="Train"):
  filenames = os.listdir(dataset)
  data_dir = os.path.join(os.sep, os.getcwd(), dataset)
  data = glob.glob(os.path.join(os.sep, data_dir, "*.bmp"))

  return data

def make_data(data, label):
  savepath = os.path.join(os.sep, os.getcwd(), 'checkpoint/train.h5')
  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path, is_grayscale=True):
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def load_sample(sess, data_idx):
  data = prepare_data()

  sub_sample_sequence = []
  sub_label_sequence = []

  sample, label = preprocess(data[data_idx], FLAGS.scale)
  padding = abs(FLAGS.image_size - FLAGS.label_size) / 2

  if len(sample.shape) == 3:
    h, w, _ = sample.shape
  else:
    h, w = sample.shape

  for x in range(0, h-FLAGS.image_size+1, FLAGS.stride):
    for y in range(0, w-FLAGS.image_size+1, FLAGS.stride):
      sub_sample = sample[x:x+FLAGS.image_size, y:y+FLAGS.image_size]
      sub_label = label[x+padding:x+padding+FLAGS.label_size, y+padding:y+padding+FLAGS.label_size]

      sub_sample = sub_sample.reshape([FLAGS.image_size, FLAGS.image_size, 1])  
      sub_label = sub_label.reshape([FLAGS.label_size, FLAGS.label_size, 1])

      sub_sample_sequence.append(sub_sample)
      sub_label_sequence.append(sub_label)

  arrsample = np.asarray(sub_sample_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  return arrsample, arrlabel

def input_setup(sess):
  data = prepare_data()
  
  sub_input_sequence = []
  sub_label_sequence = []
  
  padding = abs(FLAGS.image_size - FLAGS.label_size) / 2

  for i in xrange(len(data)):
    input_, label_ = preprocess(data[i], FLAGS.scale)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    for x in range(0, h-FLAGS.image_size+1, FLAGS.stride):
      for y in range(0, w-FLAGS.image_size+1, FLAGS.stride):
        sub_input = input_[x:x+FLAGS.image_size, y:y+FLAGS.image_size]
        sub_label = label_[x+padding:x+padding+FLAGS.label_size, y+padding:y+padding+FLAGS.label_size]

        sub_input = sub_input.reshape([FLAGS.image_size, FLAGS.image_size, 1])  
        sub_label = sub_label.reshape([FLAGS.label_size, FLAGS.label_size, 1])

        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

  make_data(arrdata, arrlabel)

def psnr(img1, img2):
  mse = np.mean((img1 - img2) ** 2)
  if mse == 0:
    return 100
  PIXEL_MAX = 255.
  return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))