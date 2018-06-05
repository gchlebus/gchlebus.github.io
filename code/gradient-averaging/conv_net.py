# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np

class ConvNet(object):
  def __init__(self, dropout=0.5):
    self._input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    self._training = tf.placeholder_with_default(False, shape=[])
    self._labels = tf.placeholder(tf.float32, shape=[None, 10])

    out = self._input
    if dropout:
      out = tf.layers.dropout(out, rate=0.2 if dropout > 0.2 else dropout, training=self._training)
    out = self.conv2d(out, filters=6, name='layer0') # 24
    out = tf.layers.max_pooling2d(out, pool_size=2, strides=2) # 12
    if dropout:
      out = tf.layers.dropout(out, rate=dropout, training=self._training)
    out = self.conv2d(out, filters=16, name='layer1') # 8

    out = tf.layers.max_pooling2d(out, pool_size=2, strides=2) # 4
    if dropout:
      out = tf.layers.dropout(out, rate=dropout, training=self._training)
    out = tf.contrib.layers.flatten(out)
    out = self.dense(out, filters=120, name='layer2')
    if dropout:
      out = tf.layers.dropout(out, rate=dropout, training=self._training)
    out = self.dense(out, filters=84, name='layer3')
    if dropout:
      out = tf.layers.dropout(out, rate=dropout, training=self._training)
    out = self.dense(out, filters=10, name='layer4')
    self._inference_op = out

    correct = tf.equal(tf.argmax(self._labels, 1), tf.argmax(self._inference_op, 1))
    self._accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))

  @classmethod
  def conv2d(cls, input, filters, name=None):
    return tf.layers.conv2d(input, filters=filters, kernel_size=5, activation=tf.nn.relu, name=name)

  @classmethod
  def dense(cls, input, filters, name=None):
    return tf.layers.dense(input, filters, activation=tf.nn.relu, name=name)

  def setup_train(self, average_gradients=1, lr=1e-3):
    self._average_gradients = average_gradients
    self._loss_op = tf.losses.softmax_cross_entropy(onehot_labels=self._labels, logits=self._inference_op)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    if average_gradients == 1:
      # This 'train_op' computes gradients and applies them in one step.
      self._train_op = optimizer.minimize(self._loss_op)
    else:
      # here 'train_op' only applies gradients passed via placeholders stored
      # in 'grads_placeholders. The gradient computation is done with 'grad_op'.
      grads_and_vars = optimizer.compute_gradients(self._loss_op)
      avg_grads_and_vars = []
      self._grad_placeholders = []
      for grad, var in grads_and_vars:
        grad_ph = tf.placeholder(grad.dtype, grad.shape)
        self._grad_placeholders.append(grad_ph)
        avg_grads_and_vars.append((grad_ph, var))
      self._grad_op = [x[0] for x in grads_and_vars]
      self._train_op = optimizer.apply_gradients(avg_grads_and_vars)
      self._gradients = [] # list to store gradients

  def train(self, session, input_batch, output_batch):
    feed_dict = {
      self._input: input_batch,
      self._labels: output_batch,
      self._training: True
    }
    if self._average_gradients == 1:
      loss, _ = session.run([self._loss_op, self._train_op], feed_dict=feed_dict)
    else:
      loss, grads = session.run([self._loss_op, self._grad_op], feed_dict=feed_dict)
      self._gradients.append(grads)
      if len(self._gradients) == self._average_gradients:
        for i, placeholder in enumerate(self._grad_placeholders):
          feed_dict[placeholder] = np.stack([g[i] for g in self._gradients], axis=0).mean(axis=0)
        session.run(self._train_op, feed_dict=feed_dict)
        self._gradients = []
    return loss

  def evaluate(self, session, input_batch, output_labels):
    feed_dict = {
      self._input: input_batch,
      self._labels: output_labels
    }
    return session.run(self._accuracy_op, feed_dict=feed_dict) * 100


