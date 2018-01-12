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
        correspondence_candidates = determine_correspondences(idx, testarr, refarr)
        if not correspondence_candidates:
            continue
        similarities = []
        for ref_indices, test_indicies in correspondence_candidates:
            current_refarr = filter_tumors(refarr, ref_indices)
            current_testarr = filter_tumors(testarr, test_indicies)
            similarities.append(similarity(current_testarr > 0, current_refarr > 0, similarity_measure))
        max_s = max(similarities)
        max_idx = [i for i, s in enumerate(similarities) if s == max_s]
        if max_s > tp_threshold:
            ref_indices, test_indicies = correspondence_candidates[max_idx[0]]
            current_refarr = filter_tumors(refarr, ref_indices)
            current_testarr = filter_tumors(testarr, test_indicies)
            tp += len(ref_indices)
            correspondences.append([ref_indices.tolist(), test_indicies.tolist()])
            tp_similarities.append(max_s)
            testarr[current_testarr > 0] = 0
            refarr[current_refarr > 0] = 0
    return tp, len(unique(testarr)), correspondences, tp_similarities, unique(testarr).tolist()

def filter_tumors(arr, tumor_indices):
    '''
    Returns array with tumors specified by tumor_indices.
    '''
    retarr = np.zeros_like(arr)
    for i in tumor_indices:
        retarr[arr==i] = i
    return retarr

def determine_correspondences(testidx, testarr, refarr):
    '''
    Determine tumor correspondences between tumors in the testarr and refarr. The correspondence
    algorithm starts with a tumor with index textidx from the testarr.
    Returns list with correspondence candidates.
    '''
    current_testarr = np.where(testarr == testidx, testidx, 0)
    ref_components_count = 0
    correspondences = []
    while True:
        current_refarr = get_overlapping_mask(np.where(current_testarr > 0, 1, 0), refarr)

        unique_ids = unique(current_refarr)
        count = len(unique_ids)
        if count == ref_components_count:
            break
        else:
            ref_components_count = count
        from itertools import combinations
        for r in range(len(unique_ids)):
            for c in combinations(unique_ids, r):
                correspondences.append([np.asarray(c), unique(current_testarr)])
        current_testarr = get_overlapping_mask(np.where(current_refarr > 0, 1, 0), testarr)
    if ref_components_count > 0:
        correspondences.append([unique(current_refarr), unique(current_testarr)])
    return correspondences


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
    testarr = testarr.astype(np.int32)
    refarr = refarr.astype(np.int32)
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
