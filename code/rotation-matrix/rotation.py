# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import numpy as np

def rotation_matrix(axis, angle):
  '''
  Returns rotation matrix around *axis* with *angle* radians.
  >>> rotation_matrix([1, 0, 0], 0)
  array([[ 1.,  0.,  0.,  0.],
         [ 0.,  1.,  0.,  0.],
         [ 0.,  0.,  1.,  0.],
         [ 0.,  0.,  0.,  1.]])
  >>> rotation_matrix([0, 0, 1], np.radians(158.17))
  array([[ 1.,  0.,  0.,  0.],
         [ 0.,  1.,  0.,  0.],
         [ 0.,  0.,  1.,  0.],
         [ 0.,  0.,  0.,  1.]])
  >>> rotation_matrix([1, 0, 0], np.radians(-0.01))
  array([[ 1.,  0.,  0.,  0.],
         [ 0.,  1.,  0.,  0.],
         [ 0.,  0.,  1.,  0.],
         [ 0.,  0.,  0.,  1.]])
  >>> rotation_matrix([1, 0, 0], np.radians(-90))
  array([[ 1.,  0.,  0.,  0.],
         [ 0.,  1.,  0.,  0.],
         [ 0.,  0.,  1.,  0.],
         [ 0.,  0.,  0.,  1.]])
  '''
  c = np.cos(angle)
  s = np.sin(angle)
  n1, n2, n3 = axis
  return np.array(
    [
      [n1*n1*(1-c)+c, n1*n2*(1-c)-n3*s, n1*n3*(1-c)+n2*s, 0],
      [n2*n1*(1-c)+n3*s, n2*n2*(1-c)+c, n2*n3*(1-c)-n1*s, 0],
      [n3*n1*(1-c)-n2*s, n3*n2*(1-c)+n1*s, n3*n3*(1-c)+c, 0],
      [0, 0, 0, 1],
    ]
  )

if __name__ == '__main__':
  #import doctest
  #doctest.testmod()
  rot = rotation_matrix([1, 0, 0], np.radians(-0.01))
  for i in range(rot.shape[0]):
    print(rot[i])