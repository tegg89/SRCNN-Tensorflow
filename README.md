# SRCNN-Tensorflow
Tensorflow implementation of Convolutional Neural Networks for super-resolution. The original Matlab and Caffe from official website can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).

## Prerequisites
 * Tensorflow
 * Scipy version > 0.18 ('mode' option from scipy.misc.imread function)
 * h5py
 * matplotlib

This code requires Tensorflow. Also scipy is used instead of Matlab or OpenCV. Especially, installing OpenCV at Linux is sort of complicated. So, with reproducing this paper, I used scipy instead. For more imformation about scipy, click [here](https://www.scipy.org/).

## Usage
For training, `python main.py`
<br>
For testing, `python main.py --is_train False --stride 21`

## Result
Original butterfly image:
<img src="https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/orig.png" width="250" height="250" />
Bicubic interpolated image:
<img src="https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/bicubic.png" width="250" height="250" />
Super-resolved image:
<img src="https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/srcnn.png" width="250" height="250" />

## References
* [liliumao/Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn) 
  * - I referred to this repository which is same implementation using Matlab code and Caffe model.
<br>
* [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) 
  * - I have followed and learned training process and structure of this repository.

