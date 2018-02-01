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

if __name__ == '__main__':
  args = parse_args()
  model = UNet()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  input_shape = [1] + UNet.INPUT_SHAPE
  input_batch = np.random.rand(*input_shape)
  print('Class distribution bg: %d%% fg: %d%%' % (args.t*100, (1-args.t)*100))
  output_batch = np.concatenate([input_batch<=args.t, input_batch>args.t], axis=-1).astype(np.float32)
  for i in range(args.i):
    _, cce_loss, cce_grad, dice_fg_loss, dice_fg_grad, dice_loss, dice_grad = model.train(sess, input_batch, output_batch)
    print('[%d] cce_loss: %.3g cce_norm: %.3g dice_fg_loss: %.3g dice_fg_norm: %.3g dice_loss: %.3g dice_norm: %.3g' % 
      (i, cce_loss, cce_grad, dice_fg_loss, dice_fg_grad, dice_loss, dice_grad))