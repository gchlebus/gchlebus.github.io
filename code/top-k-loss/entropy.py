import tensorflow as tf
import numpy as np
import click

NUM_CLASSES = 3

#TODO compare the keras implementations!

"""
Conclusions
- CCE requires softmax activation and is for multi-class problems
  yields the same results as BCE for problems with only one fg class.
  Here only the probability of the true classes is used to compute the loss. 
  Probabilities of other classes are influenced via the softmax function.
- BCE doesn't make any assumptions regarding the activation function.
  BCE has to be used for mulit-label problems.
  BCE considers class A vs. non class A
"""

def categorical_cross_entropy(y, y_target):
  y /= tf.reduce_sum(y, -1, True)
  logProbabilities = tf.log(tf.clip_by_value(y, 1e-7, 1.0-1e-7))
  #loss = -tf.reduce_sum(y_target * logProbabilities, axis=-1)
  loss = -y_target * logProbabilities
  #return loss
  return tf.reduce_mean(loss)

def binary_cross_entropy(y, y_target):
  distance = tf.abs(y - y_target)
  logDistance = -tf.log(1 - tf.clip_by_value(distance, 1e-7, 1.0 - 1e-7))
  #return logDistance
  return tf.reduce_mean(logDistance)

def keras_binary(output, target, from_logits=False):
  if not from_logits:
      # transform back to logits
      _epsilon = 1e-7
      output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
      output = tf.log(output / (1 - output))
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
  #return loss
  return tf.reduce_mean(loss)

def keras_cce(output, target, from_logits=False, axis=-1):
  if not from_logits: # here a sigmoid or softmax is assumed, 
    # scale preds so that the class probas of each sample sum to 1
    output /= tf.reduce_sum(output, axis, True)
    # manual computation of crossentropy
    _epsilon = 1e-7
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    #loss = - tf.reduce_sum(target * tf.log(output), axis) 
    loss = - target * tf.log(output)
  else:
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)
  #return loss
  return tf.reduce_mean(loss)



# logits - unscaled log probabilities
def softmax(logits):
  logits -= np.max(logits, axis=-1, keepdims=True)
  exp_logits = np.exp(logits)
  return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def sigmoid(logits):
  return 1 / (1 + np.exp(-logits))

def get_losses_dict(output, target):
  tf.reset_default_graph()
  output_ph = tf.placeholder(tf.float32, shape=[1, len(output[0])])
  target_ph = tf.placeholder(tf.float32, shape=[1, len(output[0])])
  cce_loss = categorical_cross_entropy(output, target)
  bce_loss = binary_cross_entropy(output, target)
  keras_bce_loss = keras_binary(output_ph, target_ph)
  keras_cce_loss = keras_cce(output_ph, target_ph)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ret = sess.run([cce_loss, bce_loss, keras_cce_loss, keras_bce_loss], feed_dict={
      output_ph: output,
      target_ph: target
    })
    ret = [x.squeeze() for x in ret]
    return dict(
      categorical_cross_entropy=ret[0],
      binary_cross_entropy=ret[1],
      keras_categorical_cross_entropy=ret[2],
      keras_binary_cross_entropy=ret[3]
    )

def print_losses(output, target, activation_fn):
  output = np.asarray(output).astype(np.float32)[np.newaxis, :]
  target = np.asarray(target).astype(np.float32)[np.newaxis, :]
  if output.shape != target.shape:
    raise RuntimeError("target/output shape mismatch")

  print('Output:', output.squeeze())
  #output = activation_fn(output)
  #print('Output after %s:' % activation_fn.__name__, output.squeeze())
  print('Target:', target.squeeze())

  for loss_name, loss_output in get_losses_dict(output, target).items():
    print("%s = " % loss_name, loss_output)

if __name__ == '__main__':
  output = [0.3, 0.6, 0.1]
  target = [0, 1, 0]
  print_losses(output, target, sigmoid)