# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np
from u_net import UNet

def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', type=int, default=10)
  parser.add_argument('-t', type=float, default=0.5)
  return parser.parse_args()

def get_data(threshold):
  input_shape = [1] + UNet.INPUT_SHAPE
  input_batch = np.random.rand(*input_shape)
  output_batch = np.concatenate([input_batch<=threshold, input_batch>threshold], axis=-1).astype(np.float32)
  bg_count = np.sum(output_batch[...,0])
  fg_count = np.sum(output_batch[...,1])
  bg_percent = 100 * bg_count / (bg_count + fg_count)
  print('Class distribution bg: %.2g%% fg: %.2g%%' % (bg_percent, 100-bg_percent))
  return input_batch, output_batch

if __name__ == '__main__':
  args = parse_args()
  model = UNet(filters=4)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  input_batch, output_batch = get_data(args.t)
  for i in range(args.i):
    _, cce_loss, cce_grad, dice_fg_loss, dice_fg_grad, dice_loss, dice_grad = model.train(sess, input_batch, output_batch)
    print('[%d] cce_loss: %.3g cce_norm: %.3g dice_fg_loss: %.3g dice_fg_norm: %.3g dice_loss: %.3g dice_norm: %.3g' % 
      (i, cce_loss, cce_grad, dice_fg_loss, dice_fg_grad, dice_loss, dice_grad))