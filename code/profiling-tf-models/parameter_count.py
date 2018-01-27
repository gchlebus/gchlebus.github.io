# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np
from u_net import UNet

def parameter_count(graph):
  '''
  >>> u_net = UNet()
  >>> parameter_count(tf.get_default_graph())
  31030658
  '''
  vars = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  return (np.sum([np.prod(v.shape) for v in vars])).value

if __name__ == '__main__':
  import doctest
  doctest.testmod()
