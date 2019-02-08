import tensorflow as tf
import numpy as np

def has_channels_first_support():
  return False

def to_channels_last(tensor):
  ndims = tensor.get_shape().ndims
  if ndims == 4:
    return tf.transpose(tensor, perm=[0, 2, 3, 1])
  elif ndims == 5:
    return tf.transpose(tensor, perm=[0, 2, 3, 4, 1])

def to_channels_first(tensor):
  ndims = tensor.get_shape().ndims
  if ndims == 4:
    return tf.transpose(tensor, perm=[0, 3, 1, 2])
  elif ndims == 5:
    return tf.transpose(tensor, perm=[0, 4, 1, 2, 3])

def conv3d(input, filters, kernel_size=(3, 3, 3), strides=(1,1,1),
          padding="same", kernel_initializer=tf.keras.initializers.he_normal(),
          use_bias=True, name="conv3d"):
  with tf.variable_scope(name):
    channel_axis = 1
    kernel_shape = kernel_size + (input.shape[channel_axis], filters)
    kernel = tf.get_variable("kernel", shape=kernel_shape, dtype=tf.float32,
      initializer=kernel_initializer, trainable=True)

    out = to_channels_last(input)
    out = tf.nn.conv3d(out, kernel, (1,) + strides + (1,), padding=padding.upper())
    if use_bias:
      bias = tf.get_variable("bias", shape=(filters), dtype=tf.float32,
        initializer=tf.keras.initializers.zeros(), trainable=True)
      out = tf.add(out, bias)
    return to_channels_first(out)

def conv3d_transpose(input, filters, kernel_size=(2,2,2), strides=(2,2,2),
    padding="same", kernel_initializer=tf.keras.initializers.he_normal(),
    use_bias=True, name="conv3d_transpose"):
  with tf.variable_scope(name):
    channel_axis = 1
    kernel_shape = kernel_size + (filters, input.shape[channel_axis])
    kernel = tf.get_variable("kernel", shape=kernel_shape, dtype=tf.float32,
      initializer=kernel_initializer, trainable=True)

    out = to_channels_last(input)
    input_shape = tf.shape(out)
    output_shape = [
      input_shape[0],
      input_shape[1] * kernel_size[0],
      input_shape[2] * kernel_size[1],
      input_shape[3] * kernel_size[2],
      filters
    ]
    out = tf.nn.conv3d_transpose(out, kernel, output_shape, strides=(1,) + strides + (1,),
        padding=padding.upper())
    if use_bias:
      bias = tf.get_variable("bias", shape=(filters), dtype=tf.float32,
        initializer=tf.keras.initializers.zeros(), trainable=True)
      out = tf.add(out, bias)
    return to_channels_first(out)

def run(input):
  tf.reset_default_graph()
  #input_ph = tf.placeholder(tf.float32, shape=[None, 1, None, None, None])
  input_ph = tf.placeholder(tf.float32, shape=[2, 1, 20, 30, 10])
  conv_op = conv3d_transpose(input_ph, 32)
  #conv_op = conv_transpose(input_ph, 32, kernel_size=(2,2,2), strides=(2,2,2))
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ret = sess.run(conv_op, feed_dict={
      input_ph: input,
    })
    return ret

if __name__ == '__main__':
  input = np.ones((2, 1, 20, 30, 10), dtype=np.float32)
  print("input.shape", input.shape)
  print("input = ", input[:, 0, 0, 0, 0])
  ret = run(input)
  print("ret.shape", ret.shape)
  print("ret = ", ret[:, 0, 0, 0, 0])
  vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  for var in vars:
    print(var.name)