import tensorflow as tf
import numpy as np
import os

def get_config_proto():
  config = tf.ConfigProto()
  if os.getenv('GPU_MEMORY_FRACTION'):
    config.gpu_options.per_process_gpu_memory_fraction = float(os.getenv('GPU_MEMORY_FRACTION'))
  else:
    config.gpu_options.allow_growth = True
  return config

def has_channels_first_support():
  '''
  Currently, many operations are not supported on a CPU with the "channels_first" data format.
  The RedLeaf tensorflow backend can be forced to use always "channels_last" data format, by setting
  "DISABLE_CHANNELS_FIRST" env variable. It is useful, when saving tf models for usage with tf c++ mevislab modules.
  '''
  return False
  if int(os.getenv('DISABLE_CHANNELS_FIRST', 0)):
    return False
  with tf.Session(config=get_config_proto()) as sess:
    device_types = [x.device_type for x in device_lib.list_local_devices()]
  return 'GPU' in device_types

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

def upsample(input, scale=(2, 2), name='upsample'):
  with tf.variable_scope(name):
    input_shape = tf.shape(input)
    ndims = input.get_shape().ndims
    if ndims == 4:
      deconv = tf.nn.conv2d_transpose
      conv = tf.nn.conv2d
    elif ndims == 5:
      deconv = tf.nn.conv3d_transpose
      conv = tf.nn.conv3d


    strides_conv = (1,) * ndims

    if has_channels_first_support():
      data_format = 'NCDHW' if ndims == 5 else 'NCHW'
      filter_shape = scale + (input_shape[1], input_shape[1])
      filter_deconv = tf.fill(filter_shape, 1.)
      filter_conv = tf.cast(tf.fill(filter_shape, 1. / np.prod(scale)), tf.float32)
      strides_deconv = (1, 1) + scale

      out = deconv(value=input, filter=filter_deconv, output_shape=_get_output_shape(input, scale), strides=strides_deconv, padding="SAME",
                   data_format=data_format)
      out = conv(input=out, filter=filter_conv, strides=strides_conv, padding="SAME", data_format=data_format)
    else:
      data_format = 'NDHWC' if ndims == 5 else 'NHWC'
      filter_shape = scale + (input_shape[1], input_shape[1])
      filter_deconv = tf.fill(filter_shape, 1.)
      filter_conv = tf.cast(tf.fill(filter_shape, 1. / np.prod(scale)), tf.float32)
      strides_deconv = (1,) + scale + (1,)

      out = to_channels_last(input)
      out = deconv(value=out, filter=filter_deconv, output_shape=_get_output_shape(input, scale),
                   strides=strides_deconv, padding="SAME",
                   data_format=data_format)
      out = conv(input=out, filter=filter_conv, strides=strides_conv, padding="SAME", data_format=data_format)
      out = to_channels_first(out)
    return out

def _get_output_shape(input, scale):
  input_shape = tf.shape(input)
  ndims = input.get_shape().ndims
  if ndims == 4:
    if has_channels_first_support():
      ret = [
        input_shape[0],
        input_shape[1],
        input_shape[2] * scale[0],
        input_shape[3] * scale[1]
      ]
    else:
      ret = [
        input_shape[0],
        input_shape[2] * scale[0],
        input_shape[3] * scale[1],
        input_shape[1]
      ]
  elif ndims == 5:
    if has_channels_first_support():
      ret = [
        input_shape[0],
        input_shape[1],
        input_shape[2] * scale[0],
        input_shape[3] * scale[1],
        input_shape[4] * scale[2]
      ]
    else:
      ret = [
        input_shape[0],
        input_shape[2] * scale[0],
        input_shape[3] * scale[1],
        input_shape[4] * scale[2],
        input_shape[1]
      ]
  return tf.stack(ret)


def run(input):
  tf.reset_default_graph()
  input_ph = tf.placeholder(tf.float32, shape=[None, 1, None, None, None])
  upsample_op = upsample(input_ph, scale=(1,2, 2))
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ret = sess.run(upsample_op, feed_dict={
      input_ph: input,
    })
    return ret

if __name__ == '__main__':
  a = np.asfarray([1, 2, 3, 4])
  a = np.reshape(a, (1, 1, 2, 2, 1))
  print(a.squeeze())
  ret = run(a)
  print(ret.shape)
  print(ret[..., 1])
