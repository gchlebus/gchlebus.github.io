# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
from memory_saving_gradients import gradients_memory, gradients_speed, gradients_collection

class GradientType():
  PLAIN_ADAM = 'PLAIN_ADAM'
  TF_GRADIENTS = 'TF_GRADIENTS'
  GRADIENTS_SPEED = 'GRADIENTS_SPEED'
  GRADIENTS_MEMORY = 'GRADIENTS_MEMORY'
  GRADIENTS_COLLECTION = 'GRADIENTS_COLLECTION'
  ADAM_COMPUTE_GRADIENTS = 'ADAM_COMPUTE_GRADIENTS'

  @classmethod
  def all(cls):
    return [
        cls.PLAIN_ADAM, cls.TF_GRADIENTS, cls.GRADIENTS_SPEED, cls.GRADIENTS_MEMORY,
        cls.GRADIENTS_COLLECTION, cls.ADAM_COMPUTE_GRADIENTS
      ]


class UNet(object):
  def __init__(self, filters=32, n_conv=2, dropout=0, batch_norm=False, gradient_type=GradientType.PLAIN_ADAM):
    self._input = tf.placeholder(tf.float32, shape=[None, None, None, 1])
    self._training = tf.placeholder(tf.bool, shape=None)
    self._labels = tf.placeholder(tf.float32, shape=[None, None, None, 2])

    self._inference_op = self.build_model(self._input, filters, n_conv, dropout, batch_norm, self._training)

    with tf.variable_scope('train'):
      print('GRADIENT TYPE:', gradient_type)
      self._loss_op = tf.losses.softmax_cross_entropy(self._labels, logits=self._inference_op)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer()
        grads = None
        if gradient_type == GradientType.PLAIN_ADAM:
          self._train_op = optimizer.minimize(self._loss_op)
        elif gradient_type == GradientType.ADAM_COMPUTE_GRADIENTS:
          grads = optimizer.compute_gradients(self._loss_op)
          self._train_op = optimizer.apply_gradients(grads)
          return
        elif gradient_type == GradientType.TF_GRADIENTS:
          grads = tf.gradients(self._loss_op, tf.trainable_variables(),
            gate_gradients=True)
        elif gradient_type == GradientType.GRADIENTS_SPEED:
          grads = gradients_speed(self._loss_op, tf.trainable_variables())
        elif gradient_type == GradientType.GRADIENTS_MEMORY:
          grads = gradients_memory(self._loss_op, tf.trainable_variables(),gate_gradients=True)
        elif gradient_type == GradientType.GRADIENTS_COLLECTION:
          grads = gradients_collection(self._loss_op, tf.trainable_variables())
        if grads:
          self._train_op = optimizer.apply_gradients(grads_and_vars=list(zip(grads, tf.trainable_variables())))

  @classmethod
  def build_model(cls, input, filters, n_conv, dropout, batch_norm, training):
    with tf.variable_scope('model'):
      left0 = cls.unet_block(input, filters, n_conv, dropout, batch_norm, training, 'left0')
      tf.add_to_collection('checkpoints', left0)
      down = cls.transition_down(left0)
      level1 = cls.unet_block(down, 2*filters, n_conv, dropout, batch_norm, training, 'level1')
      tf.add_to_collection('checkpoints', level1)
      up = cls.transition_up(level1, filters)
      concat = tf.concat([up, left0], axis=-1)
      right0 = cls.unet_block(concat, filters, n_conv, dropout, batch_norm, training, 'right0')
      return tf.layers.conv2d(right0, filters=2, kernel_size=1, data_format='channels_last')

  @staticmethod
  def unet_block(input, filters=32, n_conv=2, dropout=0, batch_norm=False, training=False, name=None):
    out = input
    with tf.variable_scope(name):
      for idx in range(n_conv):
        with tf.variable_scope('conv{:d}'.format(idx)):
          if dropout:
            out = tf.layers.dropout(out, rate=dropout, training=training)
          out = tf.layers.conv2d(out, filters=filters, kernel_size=3, data_format='channels_last',
                                use_bias=not batch_norm, padding='same')
          if batch_norm:
            out = tf.layers.batch_normalization(out, training=training, axis=-1)
          out = tf.nn.relu(out)
    return out

  @staticmethod
  def transition_down(input):
    return tf.layers.max_pooling2d(input, pool_size=2, strides=2, data_format='channels_last')

  @staticmethod
  def transition_up(input, filters=32):
    return tf.layers.conv2d_transpose(input, filters=filters, kernel_size=2, strides=2,
            data_format='channels_last')

  def train(self, session, input_batch, output_batch, options=None, run_metadata=None):
    feed_dict = {
      self._input: input_batch,
      self._labels: output_batch,
      self._training: True
    }
    session.run([self._loss_op, self._train_op], feed_dict=feed_dict,
                          options=options, run_metadata=run_metadata)
