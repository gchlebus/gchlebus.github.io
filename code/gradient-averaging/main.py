# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

from conv_net import ConvNet
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)

def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description='Gradient Averaging Example.')
  parser.add_argument('-n', '--average-gradients', type=int, default=1, help='Average gradients every n iterations.')
  parser.add_argument('-b', '--batch-size', type=int, default=100, help='Mini-batch size.')
  parser.add_argument('-i', '--iterations', type=int, default=50000, help='Max iteration count.')
  parser.add_argument('-r', '--reps', type=int, default=10, help='Experiment runs.')
  parser.add_argument('-v', '--verbose', action='store_true')
  return parser.parse_args()

def run_experiment(average_gradients, batch_size, iterations, verbose):
  batch_size = batch_size
  tf.reset_default_graph()
  net = ConvNet()

  validation_batch = mnist.test.images
  val_count = validation_batch.shape[0]
  validation_batch = np.reshape(validation_batch, (val_count, 28, 28, 1))
  validation_labels = mnist.test.labels

  net.setup_train(average_gradients=average_gradients)
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
        if verbose:
          print('[{:d}/{:d}] loss: {:.3g}, accuracy: {:.3g}%'.format(i+1, iterations, loss, accuracy))
    accuracy = net.evaluate(sess, validation_batch, validation_labels)
    training_log.append((accuracy, iterations))
    best = sorted(training_log, key=lambda x: x[0], reverse=True)[0]
    print('Training finished. Best accuracy: {:.3g} at iteration {:d}.'.format(best[0], best[1]))
    return best[0]

if __name__ == '__main__':
  args = parse_args()
  print('EXPERIMENT STARTED')
  print(args)
  accuracies = [run_experiment(args.average_gradients, args.batch_size, args.iterations, args.verbose)
      for i in range(args.reps)]
  print('EXPERIMENT FINISHED')
  print('mean accuracy ({:d} runs): {:.3g} +/- {:.3g}'.format(args.reps, np.mean(accuracies), np.std(accuracies)))
