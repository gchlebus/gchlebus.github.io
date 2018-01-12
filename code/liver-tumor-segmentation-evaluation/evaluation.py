# -*- coding: utf-8 -*-
__author__ = 'gchlebus'

import numpy as np


def evaluate(testarr, refarr, tp_threshold=0.2, similarity_measure='dice'):
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
    tp_threshold -- threshold to determine a true positive
    similarity_measure -- similarity measure used for true positive determination. Can be either
                          'dice' or 'jaccard'.

    Returns:
    - true positive count
    - false positive count
    - determined tumor correpondences, e.g., [
      [[1, 2], [1]],
    ], meaning that reference tumors with indices 1 and 2 were used to determine whether the test
    tumor with index 1 is a true positive. If the test passes, then the tumor would be counted as
    two true positives, since it corresponds to two tumors in the reference.
    - similarity measure values for each of the corresponding tumor sets
    - array containing all tumors classified as false positives
    '''
    tp = 0
    correspondences = []
    tp_similarities = []
    if not np.any(testarr) or not np.any(refarr):
        return tp, len(unique(testarr)), correspondences, tp_similarities, unique(testarr).tolist()

    for idx in unique(testarr):
        current_testarr, current_refarr = determine_correspondences(idx, testarr, refarr)
        if not np.any(current_testarr):
            continue
        s = similarity(current_testarr, current_refarr, similarity_measure)
        if s > tp_threshold:
            tp += len(unique(current_testarr * refarr))
            correspondences.append([unique(refarr * current_refarr).tolist(),
                                    unique(testarr * current_testarr).tolist()])
            tp_similarities.append(s)
            testarr[current_testarr == 1] = 0
            refarr[current_refarr == 1] = 0

    return tp, len(unique(testarr)), correspondences, tp_similarities, unique(testarr).tolist()


def determine_correspondences(testidx, testarr, refarr):
    '''
    Determine tumor correspondences between tumors in the testarr and refarr. The correspondence
    algorithm starts with a tumor with index textidx from the testarr.
    Returns:
    - binary array with corresponding tumors from testarr
    - binary array with corresponding tumors from refarr
    '''
    current_testarr = np.where(testarr == testidx, testidx, 0)
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

def similarity(testarr, refarr, function):
    assert not np.any(refarr > 1) or not np.any(testarr > 1), 'Only arrays with 0, 1 values are allowed.'
    if function == 'dice':
        return dice(testarr, refarr)
    elif function == 'jaccard':
        return jaccard(testarr, refarr)
    else:
        raise RuntimeError('Not supported similarity function {}.'.format(function))

def jaccard(testarr, refarr):
    intersection = float(np.sum(refarr * testarr))
    union = float(np.sum(refarr + testarr) - intersection)
    try:
        return intersection / union
    except ZeroDivisionError:
        return 1.0

def dice(testarr, refarr):
    intersection = float(np.sum(refarr * testarr))
    sum = float(np.sum(refarr + testarr))
    try:
        return 2*intersection / sum
    except ZeroDivisionError:
        return 1.0
