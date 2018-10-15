import tensorflow as tf
import numpy as np

PERCENTILE = 50

shape = (1, 1, 3, 3, 3)
ch1 = np.random.random(shape)
ch2 = np.random.random(shape) + 3
input = np.concatenate([ch1, ch2], axis=1)
print(input.shape)

per = np.percentile(input, PERCENTILE, axis=(0, 2, 3, 4))
print(per)
loss = 0
for i in range(input.shape[1]):
  tmp = np.where(input[:, i] > per[i], input[:, i], 0)
  loss += np.mean(tmp)
print("loss = ", loss)

def topkloss(tensor):
  p = tf.contrib.distributions.percentile(tensor, PERCENTILE, axis=(0,2,3,4))
  loss = 0
  for i in range(tensor.shape[1]):
    out = tf.where(tensor[:, i] > p[i], tensor[:, i], tf.zeros_like(tensor[:, i]))
    loss += tf.reduce_mean(out)
  return loss

input_ph = tf.placeholder(tf.float32, shape=[None, 2, None, None, None])
out = topkloss(input)

with tf.Session() as s:
    feed_dict = {
      input_ph: input
    }
    s.run(tf.global_variables_initializer())
    ret= s.run(out, feed_dict=feed_dict)
    print("loss = ", ret)