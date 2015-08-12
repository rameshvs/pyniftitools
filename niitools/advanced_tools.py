#!/usr/bin/env python
"""
This file contains useful tools for dealing with NifTI files and performing
more application-specific operations. Input and output files will typically be
in nifti (.nii or .nii.gz) format.

Can be used from the command line.
"""
from __future__ import print_function
from __future__ import division

import sys
import ast

import numpy as np
import nibabel as nib

def scale_intensity(input, mask, new_intensity, output):
    """
    Given nifti files input and mask, linearly rescales the intensities in
    input so that the mode intensity is new_intensity, and saves the result
    in nifti file output.
    """
    if type(new_intensity) is str:
        new_intensity = ast.literal_eval(new_intensity)
    in_nii = nib.load(input)
    mask_nii = nib.load(mask)
    mask_img = mask_nii.get_data()
    assert np.sum(np.abs(in_nii.get_affine() - mask_nii.get_affine())) < 1e-2

    in_img = in_nii.get_data()
    out_img = in_img.copy()
    if (in_img.mean() < 0):
        print("%0.1f%% of voxels were below 0" % ())
    voxels = in_img[mask_img > 0]
    voxels -= voxels.min()

    (mode, _) =  mean_shift_mode_finder(voxels.squeeze())
    print(voxels.dtype)
    print(type(mode))
    print(type(new_intensity))
    print(out_img.dtype)

    out_img[mask_img > 0] = voxels * (new_intensity / mode)

    out_nii = nib.Nifti1Image(out_img, header=in_nii.get_header(),
            affine=in_nii.get_affine())

    out_nii.to_filename(output)

def mean_shift_mode_finder(data, sigma=None, n_replicates=10,
        replication_method='percentiles', epsilon=None,
        max_iterations=100, n_bins=None):
    """
    Finds the mode of data using mean shift. Returns the best
    value and its score.

    e.g., (mean, score) = mean_shift_mode_finder(data.flatten())

    Inputs
    ------
    data : data to find the mode of (one-dimensional ndarray)
    sigma : kernel sigma (h) to be used; defaults to heuristic
    n_replicates : how many times to run
    replication_method : how to determine initialization for each replicate.
                           'percentile' (uses n_replicate percentiles)
                           'random' (uses n_replicate random valid values)
    epsilon : if the change in mode is less than this value, stop
    max_iterations : maximum number of iterations for each replicate
    n_bins : how many bins to use for the data histogram

    Adapted from 'meanShift.m' by Adrian Dalca (https://github.com/adalca/mgt/)
    """

    if sigma is None:
        # Optimal bandwidth suggested by Bowman and Azzalini ('97) p31
        # adapted from ksr.m by Yi Cao
        sigma = np.median(np.abs(data-np.median(data))) / .6745 * (4/3/data.size)**0.2
    if epsilon is None:
        # heuristic
        epsilon = sigma / 100
    if n_bins is None:
        n_bins = max(data.size / 10, 1)

    # Set up histogram
    dmin, dmax = data.min(), data.max()
    bins = np.linspace(dmin, dmax, n_bins)
    bin_size = (dmax - dmin) / (n_bins - 1)
    (data_hist, _) = np.histogram(data, bins)
    bin_centers = bins[:-1] + .5 * bin_size

    # Set up replicates
    if replication_method == 'percentiles':
        if n_replicates > 1:
            percentiles = np.linspace(0, 100, n_replicates)
        else:
            percentiles = [50]

        inits = [np.percentile(data, p) for p in percentiles]

    elif replication_method == 'random':
        inits = np.random.uniform(data.min(), data.max(), n_replicates)

    scores = np.empty(n_replicates)
    means = np.empty(n_replicates)
    iter_counts = np.zeros(n_replicates) + np.inf
    # Core algorithm
    for i in xrange(n_replicates):
        mean = inits[i]
        change = np.inf
        for j in xrange(max_iterations):
            if change < epsilon:
                break
            weights = np.exp(-.5 * ((data - mean)/sigma) ** 2)
            assert weights.sum() != 0, "Weights sum to 0; increase sigma if appropriate (current val %f)" % sigma
            mean_old = mean
            mean = np.dot(weights, data) / weights.sum()
            change = np.abs(mean_old - mean)
        means[i] = mean

        kernel = np.exp(-(bin_centers - mean)**2/(2*sigma**2))
        scores[i] = np.dot(kernel, data_hist)
        iter_counts[i] = j

    best = np.argmax(scores)
    n_good_replicates = np.sum(np.abs(means[best] - means) < sigma / 5) - 1
    print(means)
    print(sigma)
    print("%d other replicates matched the best one." % n_good_replicates)
    # out = {}
    # out['score'] = scores[best]
    # out['mean'] = means[best]
    # out['iter_count'] = iter_counts[best]
    # return out

    return (means[best], scores[best])

if __name__ == '__main__':

    help = \
"""USAGE: {fname} operation arguments

List of operations and their arguments:

scale_intensity
    scale_intensity <infile> <maskfile> <intensity> <outfile>
    scales masked data in infile so that the mode is intensity
""".format(fname=sys.argv[0])
    if len(sys.argv) < 3:
        print(help, file=sys.stderr)
        sys.exit(1)
    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == 'scale_intensity':
        if len(args) == 4:
            scale_intensity(args[0], args[1], ast.literal_eval(args[2]), args[3])
        else:
            raise ValueError
