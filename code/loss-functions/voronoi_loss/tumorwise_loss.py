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

def loss(voronoi, labels, pred):
  patch_count = tf.shape(labels)[0]
  smooth = 1e-7
  allchannels = False
  axis = (0,2,3)

  def cond1(patch_index, *args):
    return tf.less(patch_index, patch_count)

  def loop1(patch_index, sum_dice, counter):
    idx = slice(patch_index, patch_index+1)
    voronoi_patch = voronoi[idx]
    labels_patch = labels[idx]
    pred_patch = pred[idx]
    voronoi_ids, _ = tf.unique(tf.reshape(voronoi_patch, [-1]))
    _, _, sum_dice, counter, *_ = tf.while_loop(cond2, loop2, [
      tf.constant(0), voronoi_ids, sum_dice, counter, voronoi_patch, labels_patch, pred_patch
      ])
    return [tf.add(patch_index, 1), sum_dice, counter]

  def cond2(voronoi_index, voronoi_ids, *args):
    return tf.less(voronoi_index, tf.size(voronoi_ids))

  def loop2(voronoi_index, voronoi_ids, sum_dice, counter, voronoi_patch, labels_patch, pred_patch):
    mask = tf.cast(tf.equal(voronoi_patch, voronoi_ids[voronoi_index]), tf.float32)
    masked_labels = (mask * labels_patch)
    masked_pred = (mask * pred_patch)
    if not allchannels:
      masked_labels = masked_labels[:, 1:]
      masked_pred = masked_pred[:, 1:]
    true_fg = tf.reduce_sum(masked_labels, axis=axis)
    pred_fg = tf.reduce_sum(masked_pred, axis=axis)
    intersection = tf.reduce_sum(masked_labels * masked_pred, axis=axis)
    dice = (2*intersection + smooth) / (true_fg + pred_fg + smooth)
    dice = tf.reduce_sum(dice)
    dice = tf.Print(dice, [dice], message='dice=')
    return tf.add(voronoi_index, 1), voronoi_ids, tf.add(sum_dice, dice), tf.add(counter, 1), voronoi_patch, labels_patch, pred_patch

  _, sum_dice, counter, *_ = tf.while_loop(cond1, loop1, [
    tf.constant(0), tf.constant(0, tf.float32), tf.constant(0, tf.float32)
  ])
  return sum_dice / counter

if __name__ == '__main__':
  voronoi, labels, pred = load_data()
  print('voronoi.shape', voronoi.shape)
  print('labels.shape', labels.shape)
  print('pred.shape', pred.shape)

  voronoi_ph = tf.placeholder(tf.float32, shape=[None, 1, None, None])
  labels_ph = tf.placeholder(tf.float32, shape=[None, 2, None, None])
  pred_ph = tf.placeholder(tf.float32, shape=[None, 2, None, None])

  final_dice = loss(voronoi_ph, labels_ph, pred_ph)
  with tf.Session() as s:
    feed_dict = {
      voronoi_ph: voronoi,
      labels_ph: labels,
      pred_ph: pred
    }
    s.run(tf.global_variables_initializer())
    ret= s.run(final_dice, feed_dict=feed_dict)
    print('First run', ret)
    ret= s.run(final_dice, feed_dict=feed_dict)
    print('Second run', ret)