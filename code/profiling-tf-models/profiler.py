# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.core.protobuf import rewriter_config_pb2
import numpy as np
from u_net import UNet, GradientType
import mem_util

class OutputType():
  FILE = 0
  TIMELINE = 1
  STDOUT = 2
  NONE = 3

def get_session(disable_optimizer):
  print('DISABLE OPTIMIZER:', str(disable_optimizer).upper())
  if disable_optimizer:
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
    config = tf.ConfigProto(operation_timeout_in_ms=150000, graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
    config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.place_pruned_graph = True
    return tf.Session(config=config)
  else:
    return tf.Session()


def profile(img_size=128, batch_size=1, filters=64, n_conv=5, dropout=0.5, batch_norm=False,
  output_type=OutputType.NONE, gradient_type=GradientType.PLAIN_ADAM, disable_optimizer=False):
  tf.reset_default_graph()
  unet = UNet(filters, n_conv, dropout, batch_norm, gradient_type=gradient_type)
  sess = get_session(disable_optimizer)
  
  sess.run(tf.global_variables_initializer())
  input_batch = np.random.rand(batch_size, img_size, img_size, 1)
  output_batch = np.random.rand(batch_size, img_size, img_size, 2)

  run_metadata = tf.RunMetadata()
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  unet.train(sess, input_batch, output_batch, options=options, run_metadata=run_metadata)

  builder = (tf.profiler.ProfileOptionBuilder()
    .with_max_depth(6)
    .select(['micros', 'bytes', 'params', 'float_ops'])
    .order_by('peak_bytes'))
  if output_type == OutputType.FILE:
    builder = builder.with_file_output('profile_output')
  elif output_type == OutputType.TIMELINE:
    builder = builder.with_timeline_output('timeline_output')
  elif output_type == OutputType.STDOUT:
    builder = builder.with_stdout_output()
  else:
    builder = builder.with_empty_output()
  options = builder.build()
  result = tf.profiler.profile(tf.get_default_graph(), run_meta=run_metadata, cmd="scope",
                      options=options)
  print(mem_util.peak_memory(run_metadata))
  #print(run_metadata)
  #tl = timeline.Timeline(run_metadata.step_stats)
  #print(tl.generate_chrome_trace_format(show_memory=True))
  #trace_file = tf.gfile.Open(name='timeline_out', mode='w')
  #trace_file.write(tl.generate_chrome_trace_format(show_memory=True))

  ret = dict()
  #print(result)
  ret['total'] = get_profiling_results(result)
  for name in 'train model'.split():
    child = [c for c in result.children if c.name == name][0]
    ret[name] = get_profiling_results(child)
  return ret

def get_profiling_results(node):
  ret = dict()
  for name in 'total_peak_bytes total_parameters'.split():
    ret[name] = getattr(node, name)
  return ret
