# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np
from conv_net import ConvNet
from tqdm import tqdm

def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', type=int, default=10)
  parser.add_argument('-t', type=float, default=0.5)
  return parser.parse_args()

def get_data(threshold, size=1, verbose=False):
  input_shape = [size] + ConvNet.INPUT_SHAPE
  input_batch = np.random.rand(*input_shape)
  output_batch = np.concatenate([input_batch<=threshold, input_batch>threshold], axis=-1).astype(np.float32)
  if verbose:
    bg_count = np.sum(output_batch[...,0])
    fg_count = np.sum(output_batch[...,1])
    bg_percent = 100 * bg_count / (bg_count + fg_count)
    print('Class distribution bg: %.2g%% fg: %.2g%%' % (bg_percent, 100-bg_percent))
  return input_batch, output_batch

def experiment(threshold, iterations, train_loss, n_conv, optimizer, batch_size=1, batch_norm=False, learning_rate=1e-3, summary_dir=None):
  model = ConvNet(filters=4, n_conv=n_conv, train_loss=train_loss, batch_norm=batch_norm, optimizer=optimizer, learning_rate=learning_rate,
    summary_dir=summary_dir)
  print('train_loss:', train_loss.value, 'optimizer:', optimizer.value,
    'n_conv:', n_conv, 'batch_norm:', batch_norm, 'batch_size:', batch_size, 'learning_rate:', learning_rate)
  ret = dict()
  val_input_batch, val_output_batch = get_data(threshold, 100, verbose=True)
  best_accuracy = (0.0, 0)
  for i in tqdm(range(iterations)):
    input_batch, output_batch = get_data(threshold, batch_size)
    out = model.train(input_batch, output_batch)
    if i == 0:
      for k in out.keys():
        ret[k] = []
      ret['accuracy'] = []
    for k,v in out.items():
      ret[k].append(v)

    if i % 250 == 0:
      accuracy = model.accuracy(val_input_batch, val_output_batch)
      if accuracy > best_accuracy[0]:
        best_accuracy = (accuracy, i)
      ret['accuracy'].append((i, accuracy))
      #print('[%d] accuracy: %.3g' % (i, accuracy))
  print('Best accuracy %.3g at iteration %d.' % best_accuracy)
  return ret
