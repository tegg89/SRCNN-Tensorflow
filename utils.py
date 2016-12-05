import scipy.misc
import os
import glob
import h5py
import PIL  # for loading images as YCbCr format
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def read_data(path):
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))

    return data, label

def save_images(images, size, image_path):
  return scipy.misc.imsave(image_path, merge(images, size))

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h : j*h+h, i*w : i*w+w, :] = image

  return img

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
  savepath = os.path.join(os.sep, os.getcwd(), 'checkpoint/train_.h5')
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

def input_setup(sess):
  data = prepare_data()

  input_, label_ = preprocess(data[0], FLAGS.scale)
  padding = abs(FLAGS.image_size - FLAGS.label_size) / 2

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  sub_input_sequence = []
  sub_label_sequence = []

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

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, srcnn, config):
  values = np.arange(0, 1, 1./128)
  for idx in [random.randint(0, 99) for _ in xrange(100)]:
    print(" [*] %d" % idx)
    make_gif(sess.samples, './samples/test_gif_%s.gif' % (idx))