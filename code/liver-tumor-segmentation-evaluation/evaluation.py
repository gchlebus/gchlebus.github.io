# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import numpy as np


def evaluate(refarr, testarr, tp_threshold=0.1):
    '''
    Evaluate tumor segmentation contained in 'testarr' against reference 'refarr'. The function
    attemtps to determine correspondences between test and reference tumor segmentation in order
    to handle correctly the following mappings:
    a) 1:n -- one tumor in the reference was segmented as n separate tumors in the test array.
    b) m:1 -- m tumors in the reference were segmented as one tumor in the test array.
    c) m:n -- m tumors in the reference were segmented as n tumors in the test array.

    Arguments:
    refarr -- ndarray with the reference tumors (each tumor has a unique value)
    testarr -- ndarray with tumors to be evaluated (each tumor has a unique value)
    th -- jaccard threshold to determine a true positive

    Returns:
    - true positive count
    - false positive count
    - determined tumor correpondences, e.g., [
      [[1, 2], [1]],
    ], meaning that reference tumors with indices 1 and 2 were used to determine whether the test
    tumor with index 1 is a true positive. If the test passes, then the tumor would be counted as
    two true positives, since it corresponds to two tumors in the reference.
    - jaccard indices for each of the corresponding tumor sets
    - array containing all tumors classified as false positives
    '''
    tp = 0
    correspondences = []
    jaccard_indices = []
    if not np.any(testarr) or not np.any(refarr):
        return tp, len(unique(testarr)), correspondences, jaccard_indices, unique(testarr).tolist()

    for idx in unique(testarr):
        current_testarr, current_refarr = determine_correspondences(idx, testarr, refarr)
        j = jaccard(current_refarr, current_testarr)
        if j > tp_threshold:
            tp += len(unique(current_testarr * refarr))
            correspondences.append([unique(refarr * current_refarr).tolist(),
                                    unique(testarr * current_testarr).tolist()])
            jaccard_indices.append(j)
            testarr[current_testarr == 1] = 0
            refarr[current_refarr == 1] = 0

    return tp, len(unique(testarr)), correspondences, jaccard_indices, unique(testarr).tolist()


def determine_correspondences(testidx, testarr, refarr):
    '''
    Determine tumor correspondences between tumors in the testarr and refarr. The correspondence
    algorithm starts with a tumor with index textidx from the testarr.
    Returns:
    - binary array with corresponding tumors from testarr
    - binary array with corresponding tumors from refarr
    '''
    current_testarr = np.where(testarr == testidx, testidx, 0)
    current_refarr = np.zeros_like(refarr)
    ref_components_count = 0
    while True:
        current_refarr = get_overlapping_mask(np.where(current_testarr > 0, 1, 0), refarr)

        count = len(unique(current_refarr))
        if count == ref_components_count:
            break
        else:
            ref_components_count = count

        current_testarr = get_overlapping_mask(np.where(current_refarr > 0, 1, 0), testarr)
    return np.where(current_testarr > 0, 1, 0), np.where(current_refarr > 0, 1, 0)


def get_overlapping_mask(arr_a, arr_b, overlap=None):
    '''
    Returns all tumors from arr_b which overlap with the structure in arr_a according to the
    overlap function. The overlap function should receive two binary ndarrays and returns whether
    they overlap. The defualt overlap function tests for a non empty interesction.
    '''
    assert not np.any(arr_a > 1), 'arr_a should be a binary image'
    overlap_indices = []
    for idx in unique(arr_a * arr_b):
        struct = np.where(arr_b == idx, 1, 0)

        if not overlap:
            def overlap(arr1, arr2):
                return np.any(arr1 * arr2)

        if overlap(arr_a, struct):
            overlap_indices.append(idx)

    ret = np.zeros_like(arr_b)
    for idx in overlap_indices:
        ret[arr_b == idx] = idx
    return ret


def unique(array):
    ''' Return list of unique values in array except for 0.'''
    ret = np.unique(array)
    return ret[1:] if ret[0] == 0 else ret


def jaccard(refarr, testarr):
    assert not np.any(refarr > 1) or not np.any(testarr > 1), 'Only arrays with 0, 1 values are allowed.'
    if not np.any(refarr) or not np.any(testarr):
        return 0
    intersection = np.sum(refarr * testarr)
    if not np.any(intersection):
        return 0
    union = np.sum(refarr + testarr) - intersection
    return float(intersection) / float(union)
