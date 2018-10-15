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
def one_hot_encoding(array):
  one_hot_dimensions = (np.newaxis, ) * 3 + (slice(None),)
  labels = np.asarray(list(range(NUM_CLASSES)))
  a = (array == labels[one_hot_dimensions])
  #a[0, 0, 0, 0] = 1
  return a

def softmax(array):
  logits = array - np.max(array, axis=-1, keepdims=True)
  logits = np.exp(logits)
  softmax = logits / np.sum(logits, axis=-1, keepdims=True)
  return softmax

def cce(y, y_target):
  logProbabilities = tf.log(tf.clip_by_value(y, 1e-7, 1.0-1e-7))
  #loss = -tf.reduce_sum(y_target * logProbabilities, axis=-1)
  loss = -y_target * logProbabilities
  return loss
  return tf.reduce_mean(loss)

def binary_entropy(y, y_target):
  distance = tf.abs(y - y_target)
  logDistance = tf.log(1 - tf.clip_by_value(distance, 1e-7, 1.0 - 1e-7))
  return -logDistance
  return -tf.reduce_mean(logDistance)

def keras_binary(output, target, from_logits=False):
  if not from_logits:
      # transform back to logits
      _epsilon = 1e-7
      output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
      output = tf.log(output / (1 - output))
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
  return loss
  return tf.reduce_mean(loss)

def keras_cce(output, target, from_logits=False, axis=-1):
  if not from_logits:
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
  return loss
  return tf.reduce_mean(loss)


in_shape = (1, 1, 1, 1)
out_shape = (1, 1, 1, NUM_CLASSES)
input_t = np.random.randint(0, NUM_CLASSES, size=in_shape)
output_t = np.random.random(out_shape)
output_t = softmax(output_t)
input_one_hot = one_hot_encoding(input_t).astype(np.int32)

print("input_one_hot.shape", input_one_hot.shape)
print("input_one_hot[0, 0, 0, :]", input_one_hot[0, 0, 0, :])
print("output_t.shape", output_t.shape)
print("output_t[0, 0, 0, :]", output_t[0, 0, 0, :])



input_ph = tf.placeholder(tf.float32, shape=[None, None, None, NUM_CLASSES])
output_ph = tf.placeholder(tf.float32, shape=[None, None, None, NUM_CLASSES])


cce_loss = cce(output_ph, input_ph)
binary_loss = binary_entropy(output_ph, input_ph)
keras_binary_loss =keras_binary(output_ph, input_ph)
keras_cce = keras_cce(output_ph, input_ph)

with tf.Session() as s:
    feed_dict = {
      input_ph: input_one_hot,
      output_ph: output_t
    }
    s.run(tf.global_variables_initializer())
    ret = s.run([cce_loss, binary_loss, keras_binary_loss, keras_cce], feed_dict=feed_dict)
    for name, value in zip('cce binary keras_binary keras_cce'.split(), ret):
      print(name, value)