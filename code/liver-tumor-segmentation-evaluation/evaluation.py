# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import numpy as np

def evaluate(refmask, testmask, th=0.5):
  tp = 0
  correspondences = []
  if not np.any(testmask) or (not np.any(refmask) and np.any(testmask)):
    return tp, len(unique(testmask)), correspondences
    
  for idx in unique(testmask):
    current_test_mask = np.where(testmask == idx, 1, 0)
    #print('current_test_mask', current_test_mask)
    if not np.any(current_test_mask):
      continue
    current_ref_mask = np.zeros_like(refmask)
    for i in unique(current_test_mask * refmask):
      current_ref_mask[refmask == i] = 1
    
    for i in unique(current_ref_mask * testmask):
      current_test_mask[testmask == i] = 1
    

    #print('jaccard current_test_mask', current_test_mask)
    #print('jaccard current_ref_mask', current_ref_mask)
    if jaccard(current_ref_mask, current_test_mask) > th:
      tp += len(unique(current_test_mask * refmask))
      correspondences.append([unique(refmask * current_ref_mask).tolist(), unique(testmask * current_test_mask).tolist()])
      testmask[current_test_mask==1] = 0
      #print('testmask', testmask)
    
  return tp, len(unique(testmask)), correspondences

def unique(array):
  ret = np.unique(array)
  return ret[1:] if ret[0] == 0 else ret
     
def jaccard(refmask, testmask):
  assert not np.any(refmask > 1)
  assert not np.any(testmask > 1)
  intersection = np.sum(refmask * testmask)
  union = np.sum(refmask + testmask) - intersection
  #print('jaccard = ', intersection / union)
  return intersection / union

if __name__ == '__main__':
  import doctest
  doctest.testmod()