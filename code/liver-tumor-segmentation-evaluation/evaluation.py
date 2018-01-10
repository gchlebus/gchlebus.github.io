# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import numpy as np

def evaluate(refmask, testmask, th=0.5):
  if not np.any(testmask):
    return 0, 0
  if not np.any(refmask) and np.any(testmask):
    return 0, len(unique(testmask))

  tp = 0
  fp = 0
  for idx in unique(testmask):
    current_test_mask = np.zeros_like(testmask)
    current_test_mask[testmask == idx] = 1
    
    current_ref_mask = np.zeros_like(refmask)
    for i in unique(current_test_mask * refmask):
      current_ref_mask[refmask == i] = 1
    if jaccard(current_ref_mask, current_test_mask) > th:
      tp += len(unique(current_test_mask * refmask))
    else:
      fp += 1
  return tp, fp

def unique(array):
  ret = np.unique(array)
  return ret[1:] if ret[0] == 0 else ret
     
def jaccard(refmask, testmask):
  assert not np.any(refmask > 1)
  assert not np.any(testmask > 1)
  intersection = np.sum(refmask * testmask)
  union = np.sum(refmask + testmask) - intersection
  return intersection / union

if __name__ == '__main__':
  import doctest
  doctest.testmod()