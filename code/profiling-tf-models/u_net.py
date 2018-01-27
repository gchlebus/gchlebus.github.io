# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
from enum import Enum
from gradientcheckpointing.memory_saving_gradients import gradients_memory, gradients_speed, gradients_collection

class GradientType(Enum):
  PLAIN_ADAM = 'PLAIN_ADAM'
  TF_GRADIENTS = 'TF_GRADIENTS'
  TF_GRADIENTS_EXPERIMENTAL_ACCUMULATE_N = 'TF_GRADIENTS_EXPERIMENTAL_ACCUMULATE_N'
  #GRADIENTS_SPEED = 'GRADIENTS_SPEED' # causes AttributeError: 'NoneType' object has no attribute 'op'
  GRADIENTS_MEMORY = 'GRADIENTS_MEMORY'
  GRADIENTS_COLLECTION = 'GRADIENTS_COLLECTION'
  ADAM_COMPUTE_GRADIENTS = 'ADAM_COMPUTE_GRADIENTS'


class UNet(object):
  def __init__(self, filters=64, n_conv=2, dropout=0, batch_norm=False, gradient_type=GradientType.PLAIN_ADAM,
    predefined_shape=False):
    if predefined_shape:
      self._input = tf.placeholder(tf.float32, shape=[1, 572, 572, 1])
      self._labels = tf.placeholder(tf.float32, shape=[1, 388, 388, 2])
    else:
      self._input = tf.placeholder(tf.float32, shape=[None, None, None, 1])
      self._labels = tf.placeholder(tf.float32, shape=[None, None, None, 2])
    self._training = tf.placeholder(tf.bool, shape=None)

    self._inference_op = self.build_model(self._input, filters, n_conv, dropout, batch_norm, self._training)

    with tf.variable_scope('train'):
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
          grads = tf.gradients(self._loss_op, tf.trainable_variables(), gate_gradients=True)
        elif gradient_type == GradientType.TF_GRADIENTS_EXPERIMENTAL_ACCUMULATE_N:
          grads = tf.gradients(self._loss_op, tf.trainable_variables(), gate_gradients=True,
          aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        #elif gradient_type == GradientType.GRADIENTS_SPEED:
        #  grads = gradients_speed(self._loss_op, tf.trainable_variables())
        elif gradient_type == GradientType.GRADIENTS_MEMORY:
          grads = gradients_memory(self._loss_op, tf.trainable_variables(),gate_gradients=True)
        elif gradient_type == GradientType.GRADIENTS_COLLECTION:
          grads = gradients_collection(self._loss_op, tf.trainable_variables(), gate_gradients=True)
        if grads:
          self._train_op = optimizer.apply_gradients(grads_and_vars=list(zip(grads, tf.trainable_variables())))

  @classmethod
  def build_model(cls, input, filters, n_conv, dropout, batch_norm, training):
    with tf.variable_scope('model'):
      left0 = cls.unet_block(input, filters, n_conv, dropout, batch_norm, training, 'left0')
      down0 = cls.transition_down(left0)
      left1 = cls.unet_block(down0, 2*filters, n_conv, dropout, batch_norm, training, 'left1')
      down1 = cls.transition_down(left1)
      left2 = cls.unet_block(down1, 4*filters, n_conv, dropout, batch_norm, training, 'left2')
      down2 = cls.transition_down(left2)
      left3 = cls.unet_block(down2, 8*filters, n_conv, dropout, batch_norm, training, 'left3')
      down3 = cls.transition_down(left3)
      across = cls.unet_block(down3, 16*filters, n_conv, dropout, batch_norm, training, 'across')
      up3 = cls.transition_up(across, 8*filters)
      concat3 = tf.concat([cls.center_crop(left3, up3), up3], axis=-1)
      right3 = cls.unet_block(concat3, 8*filters, n_conv, dropout, batch_norm, training, 'right3')
      up2 = cls.transition_up(right3, 4*filters)
      concat2 = tf.concat([cls.center_crop(left2, up2), up2], axis=-1)
      right2 = cls.unet_block(concat2, 4*filters, n_conv, dropout, batch_norm, training, 'right2')
      up1 = cls.transition_up(right2, 2*filters)
      concat1 = tf.concat([cls.center_crop(left1, up1), up1], axis=-1)
      right1 = cls.unet_block(concat1, 2*filters, n_conv, dropout, batch_norm, training, 'right1')
      up0 = cls.transition_up(right1, filters)
      concat0 = tf.concat([cls.center_crop(left0, up0), up0], axis=-1)
      right0 = cls.unet_block(concat0, filters, n_conv, dropout, batch_norm, training, 'right0')
      return tf.layers.conv2d(right0, filters=2, kernel_size=1)

  @staticmethod
  def unet_block(input, filters=32, n_conv=2, dropout=0, batch_norm=False, training=False, name=None):
    out = input
    with tf.variable_scope(name):
      for idx in range(n_conv):
        with tf.variable_scope('conv{:d}'.format(idx)):
          if dropout:
            out = tf.layers.dropout(out, rate=dropout, training=training)
          out = tf.layers.conv2d(out, filters=filters, kernel_size=3, data_format='channels_last',
                                use_bias=not batch_norm)
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

  @staticmethod
  def center_crop(input, target):
    crop = tf.cast((tf.shape(input) - tf.shape(target)) / 2, tf.int32)
    return input[:, crop[1]:-crop[1], crop[2]:-crop[2]]

  def train(self, session, input_batch, output_batch, options=None, run_metadata=None):
    feed_dict = {
      self._input: input_batch,
      self._labels: output_batch,
      self._training: True
    }
    session.run([self._loss_op, self._train_op], feed_dict=feed_dict,
                          options=options, run_metadata=run_metadata)
