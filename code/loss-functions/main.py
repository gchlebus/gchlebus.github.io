# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np
from u_net import UNet

def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', type=int, default=10)
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  model = UNet()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  input_shape = [1] + UNet.INPUT_SHAPE
  input_batch = np.random.rand(*input_shape)
  output_batch = np.concatenate([input_batch<=0.5, input_batch>0.5], axis=-1).astype(np.float32)
  for i in range(args.i):
    _, cce_loss, cce_grad, dice_fg_loss, dice_fg_grad, dice_loss, dice_grad = model.train(sess, input_batch, output_batch)
    print('[%d] cce_loss: %.3g cce_norm: %.3g dice_loss: %.3g dice_fg_norm: %.3g dice_loss: %.3g dice_norm: %.3g' % 
      (i, cce_loss, cce_grad, dice_fg_loss, dice_fg_grad, dice_loss, dice_grad))

'''
[0] cce_loss: 0.694 cce_norm: 0.0862 dice_loss: 0.502 dice_fg_norm: 0.274 dice_loss: 0.501 dice_norm: 0.043
[1] cce_loss: 0.688 cce_norm: 0.102 dice_loss: 0.499 dice_fg_norm: 0.286 dice_loss: 0.497 dice_norm: 0.0514
[2] cce_loss: 0.68 cce_norm: 0.128 dice_loss: 0.491 dice_fg_norm: 0.315 dice_loss: 0.494 dice_norm: 0.0642
[3] cce_loss: 0.668 cce_norm: 0.165 dice_loss: 0.488 dice_fg_norm: 0.391 dice_loss: 0.487 dice_norm: 0.0858
[4] cce_loss: 0.649 cce_norm: 0.222 dice_loss: 0.479 dice_fg_norm: 0.495 dice_loss: 0.477 dice_norm: 0.118
[5] cce_loss: 0.625 cce_norm: 0.289 dice_loss: 0.463 dice_fg_norm: 0.676 dice_loss: 0.464 dice_norm: 0.158
[6] cce_loss: 0.595 cce_norm: 0.36 dice_loss: 0.441 dice_fg_norm: 0.795 dice_loss: 0.447 dice_norm: 0.203
[7] cce_loss: 0.558 cce_norm: 0.427 dice_loss: 0.427 dice_fg_norm: 0.991 dice_loss: 0.425 dice_norm: 0.257
[8] cce_loss: 0.512 cce_norm: 0.484 dice_loss: 0.396 dice_fg_norm: 1.04 dice_loss: 0.397 dice_norm: 0.311
[9] cce_loss: 0.46 cce_norm: 0.531 dice_loss: 0.361 dice_fg_norm: 1.16 dice_loss: 0.362 dice_norm: 0.361
'''