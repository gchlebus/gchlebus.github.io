# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import numpy as np

def evaluate(refmask, testmask, th=0.5):
  tp = 0
  correspondences = []
  jaccard_indices = []
  if not np.any(testmask) or not np.any(refmask):
    return tp, len(unique(testmask)), correspondences, jaccard_indices, unique(testmask).tolist()
    
  for idx in unique(testmask):
    current_test_mask = np.where(testmask == idx, 1, 0)
    current_ref_mask = np.zeros_like(refmask)
    if not np.any(current_test_mask):
      continue
    
    ref_components_count = len(unique(current_ref_mask))
    while True:  
      for i in unique(current_test_mask * refmask):
        current_ref_mask[refmask == i] = i
      
      count = len(unique(current_ref_mask))
      if count == ref_components_count:
        break
      else:
        ref_components_count = count
      
      for i in unique(np.where(current_ref_mask>0, 1, 0) * testmask):
        current_test_mask[testmask == i] = 1

    current_ref_mask = np.where(current_ref_mask>0, 1, 0)
    j = jaccard(current_ref_mask, current_test_mask)
    if  j > th:
      tp += len(unique(current_test_mask * refmask))
      correspondences.append([unique(refmask * current_ref_mask).tolist(), unique(testmask * current_test_mask).tolist()])
      jaccard_indices.append(j)
      testmask[current_test_mask==1] = 0
    
  return tp, len(unique(testmask)), correspondences, jaccard_indices, unique(testmask).tolist()

def unique(array):
  ret = np.unique(array)
  return ret[1:] if ret[0] == 0 else ret
     
def jaccard(refmask, testmask):
  assert not np.any(refmask > 1)
  assert not np.any(testmask > 1)
  if not np.any(refmask) or not np.any(testmask):
    return 0
  intersection = np.sum(refmask * testmask)
  if not np.any(intersection):
    return 0
  union = np.sum(refmask + testmask) - intersection
  return intersection / union

if __name__ == '__main__':
  import doctest
  doctest.testmod()