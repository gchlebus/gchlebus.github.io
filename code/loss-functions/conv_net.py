# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np
from enum import Enum

class TrainLoss(Enum):
  CCE = 'CCE'
  DICEFG = 'DICEFG'
  DICE = 'DICE'
  DICEFG_SQUARE = 'DICEFG_SQUARE'
  DICE_SQUARE = 'DICE_SQUARE'

class ConvNet(object):
  INPUT_SHAPE = [32, 32, 1]
  OUTPUT_SHAPE = [32, 32, 2]

  def __init__(self, filters, n_conv, dropout=0, batch_norm=False, train_loss=TrainLoss.CCE, summary_dir=None):
    tf.reset_default_graph()
    self.session = None
    self._input = tf.placeholder(tf.float32, shape=[None, None, None, 1])
    self._labels = tf.placeholder(tf.float32, shape=[None, None, None, 2])
    self._training = tf.placeholder(tf.bool, shape=None)
    self._iteration = 0

    self._inference_op = self.build_model(self._input, filters, n_conv, dropout, batch_norm, self._training)
    self._outshape_op = tf.shape(self._inference_op)
    tf.summary.histogram('logits', self._inference_op)
    self._softmax_op = self.softmax(self._inference_op)
    tf.summary.histogram('probabilities', self._softmax_op)

    self._cce_loss_op = self.cce_loss(self._softmax_op, self._labels)
    self._dicefg_loss_op = self.dice_loss(self._softmax_op, self._labels, ignore_background=True, square=False)
    self._dice_loss_op = self.dice_loss(self._softmax_op, self._labels, ignore_background=False, square=False)
    self._dicefg_square_loss_op = self.dice_loss(self._softmax_op, self._labels, ignore_background=True, square=True)
    self._dice_square_loss_op = self.dice_loss(self._softmax_op, self._labels, ignore_background=False, square=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optimizer = tf.train.AdamOptimizer()

      cce_grads_and_vars = optimizer.compute_gradients(self._cce_loss_op)
      self._cce_grad_norm_op = tf.global_norm([g for g,v in cce_grads_and_vars])

      dicefg_grads_and_vars = optimizer.compute_gradients(self._dicefg_loss_op)
      self._dicefg_grad_norm_op = tf.global_norm([g for g,v in dicefg_grads_and_vars])

      dice_grads_and_vars = optimizer.compute_gradients(self._dice_loss_op)
      self._dice_grad_norm_op = tf.global_norm([g for g,v in dice_grads_and_vars])

      dicefg_square_grads_and_vars = optimizer.compute_gradients(self._dicefg_square_loss_op)
      self._dicefg_square_grad_norm_op = tf.global_norm([g for g,v in dicefg_square_grads_and_vars])

      dice_square_grads_and_vars = optimizer.compute_gradients(self._dice_square_loss_op)
      self._dice_square_grad_norm_op = tf.global_norm([g for g,v in dice_square_grads_and_vars])

      if train_loss == TrainLoss.CCE:
        grads_and_vars = cce_grads_and_vars
      elif train_loss == TrainLoss.DICEFG:
        grads_and_vars = dicefg_grads_and_vars
      elif train_loss == TrainLoss.DICE:
        grads_and_vars = dice_grads_and_vars
      elif train_loss == TrainLoss.DICEFG_SQUARE:
        grads_and_vars = dicefg_square_grads_and_vars
      elif train_loss == TrainLoss.DICE_SQUARE:
        grads_and_vars = dice_square_grads_and_vars

      self._train_op = optimizer.apply_gradients(grads_and_vars)
      self.setup_summaries(summary_dir, grads_and_vars)
    self._check_op = tf.add_check_numerics_ops()

    self.setup_accuracy()


  def setup_accuracy(self):
    labels  = tf.argmax(self._labels, axis=-1)
    predictions = tf.argmax(self._inference_op, axis=-1)
    self._accuracy_op, self._accuracy_update_op = tf.metrics.accuracy(labels, predictions, name='accuracy')
    vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy')
    self._vars_initializer = tf.variables_initializer(var_list=vars)

  def setup_summaries(self, summary_dir, grads_and_vars):
    self._summary_writer = None
    if summary_dir:
      for g, v in grads_and_vars:
        tf.summary.histogram(v.name.split(':')[0], v)
        tf.summary.histogram(v.name.split(':')[0] + '/grad', g)
      self._summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())
      self._summary_op = tf.summary.merge_all()

  @staticmethod
  def softmax(inference_op):
    logits = inference_op - tf.reduce_max(inference_op, axis=-1, keep_dims=True)
    return tf.nn.softmax(logits, dim=-1)

  @staticmethod
  def cce_loss(softmax_output, labels):
    eps = 1e-7
    log_p = -tf.log(tf.clip_by_value(softmax_output, eps, 1-eps))
    loss = tf.reduce_sum(labels * log_p, axis=-1)
    return tf.reduce_mean(loss)

  @staticmethod
  def dice_loss(softmax_output, labels, ignore_background=False, square=False):
    if ignore_background:
      labels = labels[..., 1:]
      softmax_output = softmax_output[..., 1:]
    axis = (0,1,2)
    eps = 1e-7
    nom = (2 * tf.reduce_sum(labels * softmax_output, axis=axis) + eps)
    if square:
      labels = tf.square(labels)
      softmax_output = tf.square(softmax_output)
    denom = tf.reduce_sum(labels, axis=axis) + tf.reduce_sum(softmax_output, axis=axis) + eps
    return 1 - tf.reduce_mean(nom / denom)

  @classmethod
  def build_model(cls, input, filters, n_conv, dropout, batch_norm, training):
    with tf.variable_scope('model'):
      out = cls.conv_block(input, filters, n_conv, dropout, batch_norm, training, name='conv')
      out = tf.layers.conv2d(out, filters=2, kernel_size=1)
      return out

  @staticmethod
  def conv_block(input, filters=32, n_conv=2, dropout=0, batch_norm=False, training=False, name=None):
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

  def inference(self, session, input_batch):
    feed_dict = {
      self._input: input_batch,
      self._training: True
    }
    return session.run(self._inference_op, feed_dict=feed_dict)

  def train(self, input_batch, output_batch):
    self.ensure_session()
    self.session.run(self._vars_initializer)

    feed_dict = {
      self._input: input_batch,
      self._labels: output_batch,
      self._training: True
    }
    _, _, cce_loss, cce_grad, dicefg_loss, dicefg_grad, dice_loss, dice_grad, \
    dicefg_square_loss, dicefg_square_grad, dice_square_loss, dice_square_grad = \
      self.session.run([self._check_op, self._train_op,
        self._cce_loss_op, self._cce_grad_norm_op,
        self._dicefg_loss_op, self._dicefg_grad_norm_op,
        self._dice_loss_op, self._dice_grad_norm_op,
        self._dicefg_square_loss_op, self._dicefg_square_grad_norm_op,
        self._dice_square_loss_op, self._dice_square_grad_norm_op
      ], feed_dict=feed_dict)

    if self._summary_writer:
      summary = self.session.run(self._summary_op, feed_dict=feed_dict)
      self._summary_writer.add_summary(summary, global_step=self._iteration)
      self._summary_writer.flush()

    self._iteration += 1
    return {
      'cce_loss': cce_loss, 'cce_grad': cce_grad,
      'dicefg_loss': dicefg_loss, 'dicefg_grad': dicefg_grad,
      'dice_loss': dice_loss, 'dice_grad': dice_grad,
      'dicefg_square_loss': dicefg_square_loss, 'dicefg_square_grad': dicefg_square_grad,
      'dice_square_loss': dice_square_loss, 'dice_square_grad': dice_square_grad,
    }

  def get_output_shape(self, input_shape=None):
    self.ensure_session()
    if input_shape is None:
      return self._inference_op.shape

    feed_dict = {
      self._input: np.ones([1,] + input_shape),
      self._training: False
    }
    return self.session.run(self._outshape_op, feed_dict=feed_dict)

  def accuracy(self, input_batch, output_batch):
    self.ensure_session()
    self.session.run(self._vars_initializer)

    feed_dict = {
      self._input: input_batch,
      self._labels: output_batch,
      self._training: True
    }
    self.session.run(self._accuracy_update_op, feed_dict=feed_dict)
    return self.session.run(self._accuracy_op)

  def ensure_session(self):
    if not self.session:
      self.session = tf.Session()
      self.session.run(tf.global_variables_initializer())

if __name__ == '__main__':
  model = ConvNet(4,1)
  print(model.get_output_shape(None))
  print(model.get_output_shape(input_shape=[100, 100, 1]))
