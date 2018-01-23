# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import tensorflow as tf
import numpy as np
from u_net import UNet, GradientType
from profiler import profile, OutputType
import pprint
import os



if __name__ == '__main__':
  ret = {}
  for gradientType in GradientType:
    ret[gradientType.value] = profile(gradient_type=gradientType, disable_optimizer=False)
  pprint.pprint(ret)

