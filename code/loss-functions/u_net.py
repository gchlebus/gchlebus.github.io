# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf

class UNet(object):
  INPUT_SHAPE = [572, 572, 1]
  OUTPUT_SHAPE = [572, 572, 2]

  def __init__(self, filters=64, dropout=0, batch_norm=False):
    tf.reset_default_graph()
    self._input = tf.placeholder(tf.float32, shape=[None, None, None, 1])
    self._labels = tf.placeholder(tf.float32, shape=[None, None, None, 2])
    self._training = tf.placeholder(tf.bool, shape=None)

    self._inference_op = self.build_model(self._input, filters, 2, dropout, batch_norm, self._training)
    self._probabilities_op = self.softmax(self._inference_op)

    self._cce_loss_op = self.cce_loss(self._probabilities_op, self._labels)
    self._dicefg_loss_op = self.dice_loss(self._probabilities_op, self._labels, fg_only=True)
    self._dice_loss_op = self.dice_loss(self._probabilities_op, self._labels, fg_only=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optimizer = tf.train.AdamOptimizer()

      grads_and_vars = optimizer.compute_gradients(self._cce_loss_op)
      self._cce_grad_norm_op = tf.global_norm([g for g,v in grads_and_vars])

      grads_and_vars = optimizer.compute_gradients(self._dicefg_loss_op)
      self._dicefg_grad_norm_op = tf.global_norm([g for g,v in grads_and_vars])

      grads_and_vars = optimizer.compute_gradients(self._dice_loss_op)
      self._dice_grad_norm_op = tf.global_norm([g for g,v in grads_and_vars])
      self._train_op = optimizer.apply_gradients(grads_and_vars)

  @staticmethod
  def softmax(inference_op):
    logits = inference_op - tf.reduce_max(inference_op, axis=-1, keep_dims=True)
    return tf.nn.softmax(logits, axis=-1)

  @staticmethod
  def cce_loss(probabilities, labels):
    eps = 1e-7
    log_p = -tf.log(tf.clip_by_value(probabilities, eps, 1-eps))
    loss = tf.reduce_sum(labels * log_p, axis=-1)
    return tf.reduce_mean(loss)

  @staticmethod
  def dice_loss(probabilities, labels, fg_only=False):
    if fg_only:
      true_fg = labels[..., 1:]
      pred_fg = probabilities[..., 1:]
    else:
      true_fg = labels
      pred_fg = probabilities
    axis = (0,1,2)
    tp = tf.reduce_sum(true_fg, axis=axis)
    pp = tf.reduce_sum(pred_fg, axis=axis)
    intersection = tf.reduce_sum(pred_fg * true_fg, axis=axis)
    loss = 1 - tf.reduce_mean((2 * intersection) / (tp + pp))
    return loss

  @classmethod
  def build_model(cls, input, filters, n_conv, dropout, batch_norm, training):
    with tf.variable_scope('model'):
      left0 = cls.unet_block(input, filters, n_conv, dropout, batch_norm, training, 'left0')
      down0 = cls.transition_down(left0)
      left1 = cls.unet_block(down0, 2*filters, n_conv, dropout, batch_norm, training, 'left1')
      down1 = cls.transition_down(left1)
      across = cls.unet_block(down1, 4*filters, n_conv, dropout, batch_norm, training, 'across')
      up1 = cls.transition_up(across, 2*filters)
      with tf.variable_scope('concat1'):
        concat1 = tf.concat([left1, up1], axis=-1)
      right1 = cls.unet_block(concat1, 2*filters, n_conv, dropout, batch_norm, training, 'right1')
      up0 = cls.transition_up(right1, filters)
      with tf.variable_scope('concat0'):
        concat0 = tf.concat([left0, up0], axis=-1)
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

  @staticmethod
  def center_crop(input, target):
    crop = tf.cast((tf.shape(input) - tf.shape(target)) / 2, tf.int32)
    return input[:, crop[1]:-crop[1], crop[2]:-crop[2]]

  def inference(self, session, input_batch):
    feed_dict = {
      self._input: input_batch,
      self._training: True
    }
    return session.run(self._inference_op, feed_dict=feed_dict)

  def train(self, session, input_batch, output_batch):
    feed_dict = {
      self._input: input_batch,
      self._labels: output_batch,
      self._training: True
    }
    return session.run([self._train_op,
      self._cce_loss_op, self._cce_grad_norm_op,
      self._dicefg_loss_op, self._dicefg_grad_norm_op,
      self._dice_loss_op, self._dice_grad_norm_op
      ], feed_dict=feed_dict)
