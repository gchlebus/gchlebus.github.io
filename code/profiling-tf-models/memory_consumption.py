# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np
from u_net import UNet
from gradientcheckpointing.test import mem_util

def peak_memory(model, batch_size=1, mode='train'):
  '''
  >>> peak_memory(UNet(), mode='train')
  {'/cpu:0': 2093226660}
  >>> peak_memory(UNet(), mode='inference')
  {'/cpu:0': 180936736}
  '''
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  input_shape = [batch_size] + UNet.INPUT_SHAPE
  output_shape = [batch_size] + UNet.OUTPUT_SHAPE
  input_batch = np.random.rand(*input_shape)
  output_batch = np.random.rand(*output_shape)
  run_metadata = tf.RunMetadata()
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  if mode == 'train':
    model.train(sess, input_batch, output_batch, options, run_metadata)
  elif mode == 'inference':
    model.inference(sess, input_batch, options, run_metadata)
  return mem_util.peak_memory(run_metadata)

if __name__ == '__main__':
  import doctest
  doctest.testmod()