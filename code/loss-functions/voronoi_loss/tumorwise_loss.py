# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np

def load_data():
  data = np.load('images.npz')
  voronoi = data['voronoi'][:, np.newaxis]
  labels = data['labels'][:, np.newaxis]
  labels = np.concatenate([labels==0, labels==1], axis=1).astype(np.float32)
  pred = data['prediction'][:, np.newaxis]
  pred = np.concatenate([pred==0, pred==1], axis=1).astype(np.float32)
  return voronoi, labels, pred


def cond(i, iter_count, *args):
  return tf.less(i, iter_count)

def body(i, iter_count, dicesum, unique_ids, voronoi, labels, pred):
  idx = unique_ids[i]
  mask = tf.cast(tf.equal(voronoi, idx), tf.float32)
  #mask = tf.Print(mask, [tf.shape(mask)], message='mask:', summarize=20)
  masked_labels = (labels * mask)[:, 1:]
  masked_pred = (pred * mask)[:, 1:]

  true_fg = tf.reduce_sum(masked_labels, axis=(0,2,3))
  pred_fg = tf.reduce_sum(masked_pred, axis=(0,2,3))
  intersection = tf.reduce_sum(masked_labels * masked_pred, axis=(0,2,3))

  eps = 0
  dice = (2*intersection + eps) / (true_fg + pred_fg + eps)
  dice = tf.reduce_sum(dice)
  dice = tf.Print(dice, [dice], message='dice=')
  dicesum = tf.add(dicesum, dice)
  return tf.add(i, 1), iter_count, dicesum, unique_ids, voronoi, labels, pred

if __name__ == '__main__':
  voronoi, labels, pred = load_data()
  print('voronoi.shape', voronoi.shape)
  print('labels.shape', labels.shape)
  print('pred.shape', pred.shape)

  i = tf.constant(0)
  dicesum = tf.constant(0, dtype=tf.float32)
  voronoi_var = tf.Variable(voronoi)
  labels_var = tf.Variable(labels)
  pred_var = tf.Variable(pred)

  unique_ids, _ = tf.unique(tf.reshape(voronoi_var[0:1], [-1]))
  size = tf.size(unique_ids)
  _, _, dicesum1, *_ = tf.while_loop(cond, body, [i, size, dicesum, unique_ids, voronoi_var[0:1], labels_var[0:1], pred_var[0:1]])
  dicesum1 /= tf.cast(size, tf.float32)

  unique_ids, _ = tf.unique(tf.reshape(voronoi_var[1:2], [-1]))
  size = tf.size(unique_ids)
  _, _, dicesum2, *_ = tf.while_loop(cond, body, [i, size, dicesum, unique_ids, voronoi_var[1:2], labels_var[1:2], pred_var[1:2]])
  dicesum2 /= tf.cast(size, tf.float32)
  
  final_dice = tf.reduce_mean([dicesum1, dicesum2])
  with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    print('First run')
    ret= s.run(final_dice)
    print(ret)
    print('Second run')
    ret= s.run(final_dice)
    print(ret)