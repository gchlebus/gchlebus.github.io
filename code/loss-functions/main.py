# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np
from u_net import UNet
from tqdm import tqdm

def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', type=int, default=10)
  parser.add_argument('-t', type=float, default=0.5)
  return parser.parse_args()

def get_data(threshold, size=1):
  input_shape = [size] + UNet.INPUT_SHAPE
  input_batch = np.random.rand(*input_shape)
  output_batch = np.concatenate([input_batch<=threshold, input_batch>threshold], axis=-1).astype(np.float32)
  #bg_count = np.sum(output_batch[...,0])
  #fg_count = np.sum(output_batch[...,1])
  #bg_percent = 100 * bg_count / (bg_count + fg_count)
  #print('Class distribution bg: %.2g%% fg: %.2g%%' % (bg_percent, 100-bg_percent))
  return input_batch, output_batch

def experiment(threshold, iterations, train_loss, batch_norm=False, summary_dir=None):
  model = UNet(filters=4, train_loss=train_loss, batch_norm=batch_norm, summary_dir=summary_dir)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  ret = dict()
  val_input_batch, val_output_batch = get_data(threshold, 100)
  for i in tqdm(range(iterations)):
    input_batch, output_batch = get_data(threshold)
    out = model.train(sess, input_batch, output_batch)
    if i == 0:
      for k in out.keys():
        ret[k] = []
      ret['accuracy'] = []
    for k,v in out.items():
      ret[k].append(v)

    if i % 50 == 0:
      accuracy = model.accuracy(sess, val_input_batch, val_output_batch)
      ret['accuracy'].append((i, accuracy))
  return ret

if __name__ == '__main__':
  args = parse_args()