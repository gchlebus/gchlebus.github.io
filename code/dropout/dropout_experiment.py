# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np
import collections
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)

CLIP_OPS = 'CLIP_OPS'

class ConvNet(object):
  def __init__(self, dropout=0, max_norm=0, lr=1e-3):
    self._input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    self._training = tf.placeholder(tf.bool, shape=None)
    self._labels = tf.placeholder(tf.float32, shape=[None, 10])

    if max_norm:
      max_norm = (max_norm,) * 5
    else:
      max_norm = (0,) * 5

    g = tf.get_default_graph()
    out = self._input
    if dropout:
      out = tf.layers.dropout(out, rate=0.2, training=self._training)
    out = self.conv2d(out, filters=6, max_norm=max_norm[0], name='layer0') # 24
    out = tf.layers.max_pooling2d(out, pool_size=2, strides=2) # 12
    if dropout:
      out = tf.layers.dropout(out, rate=dropout, training=self._training)
    out = self.conv2d(out, filters=16, max_norm=max_norm[1], name='layer1') # 8

    out = tf.layers.max_pooling2d(out, pool_size=2, strides=2) # 4
    if dropout:
      out = tf.layers.dropout(out, rate=dropout, training=self._training)
    out = tf.contrib.layers.flatten(out)
    out = self.dense(out, filters=120, max_norm=max_norm[2], name='layer2')
    if dropout:
      out = tf.layers.dropout(out, rate=dropout, training=self._training)
    out = self.dense(out, filters=84, max_norm=max_norm[3], name='layer3')
    if dropout:
      out = tf.layers.dropout(out, rate=dropout, training=self._training)
    out = self.dense(out, filters=10, max_norm=max_norm[4], name='layer4')
    self._inference_op = out

    self._loss_op = tf.losses.softmax_cross_entropy(onehot_labels=self._labels, logits=self._inference_op)
    self._train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._loss_op)

    correct = tf.equal(tf.argmax(self._labels, 1), tf.argmax(self._inference_op, 1))
    self._accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))

  @classmethod
  def conv2d(cls, input, filters, max_norm=0, name=None):
    out = tf.layers.conv2d(input, filters=filters, kernel_size=5, activation=tf.nn.relu, name=name)
    cls.max_norm(max_norm, name)
    return out

  @classmethod
  def dense(cls, input, filters, max_norm=0, name=None):
    out = tf.layers.dense(input, filters, activation=tf.nn.relu, name=name)
    cls.max_norm(max_norm, name)
    return out

  @staticmethod
  def max_norm(max_norm, name):
    if max_norm == 0:
      return
    g = tf.get_default_graph()
    kernel = g.get_tensor_by_name('{:s}/kernel:0'.format(name))
    tf.add_to_collection(CLIP_OPS, tf.assign(kernel, tf.clip_by_norm(kernel, max_norm)))

  def train(self, session, input_batch, output_batch):
    feed_dict = {
      self._input: input_batch,
      self._labels: output_batch,
      self._training: True
    }
    clip_ops = tf.get_collection(CLIP_OPS)
    loss, _ = session.run([self._loss_op, self._train_op], feed_dict=feed_dict)
    _ = session.run(clip_ops)
    return loss

  def evaluate(self, session, input_batch, output_labels):
    feed_dict = {
      self._input: input_batch,
      self._labels: output_labels,
      self._training: False
    }
    return session.run(self._accuracy_op, feed_dict=feed_dict) * 100

  def print_kernel_norms(self):
    g = tf.get_default_graph()
    for i in range(5):
      kernel = g.get_tensor_by_name('layer{:d}/kernel:0'.format(i)).eval()
      print('[layer{:d}] kernel norm:'.format(i), np.linalg.norm(kernel))

def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description='Dropout experiment.')
  parser.add_argument('-d', '--dropout', type=float, default=0, help='Dropout drop rate.')
  parser.add_argument('--max_norm', type=int, help='Max-norm constraint on kernel weights.', default=0)
  parser.add_argument('-i', '--iterations', type=str, default=10000, help='Max iteration count.')
  parser.add_argument('-r', '--reps', type=int, default=10, help='Experiment runs.')
  parser.add_argument('-v', '--verbose', action='store_true')
  return parser.parse_args()

def run_experiment(dropout, max_norm, iterations, verbose):
  batch_size = 100
  tf.reset_default_graph()
  net = ConvNet(dropout, max_norm)

  validation_batch = mnist.test.images
  val_count = validation_batch.shape[0]
  validation_batch = np.reshape(validation_batch, (val_count, 28, 28, 1))
  validation_labels = mnist.test.labels

  training_log = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
      batch = mnist.train.next_batch(batch_size)
      input_batch = np.reshape(batch[0], (batch_size, 28, 28, 1))
      loss = net.train(sess, input_batch, batch[1])
      if (i+1) % 100 == 0:
        accuracy = net.evaluate(sess, validation_batch, validation_labels)
        training_log.append((accuracy, i+1))
        if args.verbose:
          print('[{:d}/{:d}] loss: {:.3g}, accuracy: {:.3g}%'.format(i+1, iterations, loss, accuracy))
        net.print_kernel_norms()
    accuracy = net.evaluate(sess, validation_batch, validation_labels)
    training_log.append((accuracy, iterations))
    best = sorted(training_log, key=lambda x: x[0], reverse=True)[0]
    print('Training finished. Best accuracy: {:.3g} at iteration {:d}.'.format(best[0], best[1]))
    return best[0]

if __name__ == '__main__':
  args = parse_args()
  accuracies = [run_experiment(args.dropout, args.max_norm, args.iterations, args.verbose)
      for i in range(args.reps)]
  print('EXPERIMENT FINISHED')
  print('mean accuracy ({:d} runs): {:.3g} +/- {:.3g}'.format(args.reps, np.mean(accuracies), np.std(accuracies)))
