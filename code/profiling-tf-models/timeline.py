# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np
from u_net import UNet

def timeline(model, filename, mode='train'):
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  input_shape = [1] + UNet.INPUT_SHAPE
  output_shape = [1] + UNet.OUTPUT_SHAPE
  input_batch = np.random.rand(*input_shape)
  output_batch = np.random.rand(*output_shape)
  run_metadata = tf.RunMetadata()
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  if mode == 'train':
    model.train(sess, input_batch, output_batch, options, run_metadata)
  elif mode == 'inference':
    model.inference(sess, input_batch, options, run_metadata)
  builder = (tf.profiler.ProfileOptionBuilder()
    .with_max_depth(10)
    .select(['micros', 'bytes', 'params', 'float_ops'])
    .order_by('peak_bytes'))
  builder = builder.with_timeline_output(filename)
  options = builder.build()
  return tf.profiler.profile(tf.get_default_graph(), run_meta=run_metadata, cmd="scope",
                             options=options)

if __name__ == '__main__':
  for mode in 'train inference'.split():
    timeline(UNet(), 'timeline_{}.json'.format(mode), mode=mode)
