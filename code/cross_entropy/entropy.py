# -*- coding: utf-8 -*-

import numpy as np

def softmax(array):
  array = np.asfarray(array)
  array -= np.max(array)
  return np.exp(array) / np.sum(np.exp(array))

def cce_loss(softmax_output, target):
  softmax_output = np.asfarray(softmax_output)
  target = np.asfarray(target)
  return -target * np.log(softmax_output)

def bce_loss(output, target):
  output = np.asfarray(output)
  target = np.asfarray(target)

  d = np.abs(output - target)
  return -np.log(1 - d)
