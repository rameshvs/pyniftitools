from __future__ import print_function
from __future__ import division

import ast
import numpy as np
from numpy import newaxis as nax
import nibabel as nib

def _dice(bin1, bin2):
    """
    Computes the Dice coefficient between two binary arrays.
    """
    dice = 2*np.logical_and(bin1, bin2).sum()/(bin1.sum() + bin2.sum())
    return dice

def _jaccard(bin1, bin2):
    """
    Computes the Jaccard index between two binary arrays.
    """
    jaccard = np.logical_and(bin1, bin2).sum()/np.logical_or(bin1, bin2).sum()
    return jaccard

def majority_vote(output, *inputs):
    """
    Computes a majority vote between all inputs (assumed to be label maps) and
    saves the result in output.
    """
    niftis = [nib.load(inp) for inp in inputs]
    imgs = [nii.get_data() for nii in niftis]
    axis = imgs[0].ndim
    img_arr = np.concatenate([img[...,nax] for img in imgs], axis)

    labels = np.unique(imgs[0].flat)
    counts = np.zeros(imgs[0].shape + (len(labels),))
    for (ii,label) in enumerate(labels):
        counts[...,ii] = np.sum(img_arr == label, axis)

    header = niftis[0].get_header()
    affine = niftis[0].get_affine()
    out = labels[np.argmax(counts, axis)]
    nib.Nifti1Image(out, header=header, affine=affine).to_filename(output)

def overlap(in1, in2, out_txt, similarity, *labels):
    """
    Computes Dice/Jaccard/etc overlap between voluems in in1 and in2 using the
    provided labels. If labels are not given, assumes [1,2,3,41,42,43]

    If out_txt is given, writes output to file; otherwise prints to stdout. In
    either case, one line per label
    """
    labels = map(int, labels)
    if len(labels) == 0:
        labels = [2, 3, 4, 41, 42, 43]
    d1 = nib.load(in1).get_data()
    d2 = nib.load(in2).get_data()
    assert d1.shape == d2.shape
    out = []
    for label in labels:
        out.append(repr(similarity(d1 == label, d2 == label)))
    output = '\n'.join(out)

    if out_txt == '-':
        print(output)
    else:
        with open(out_txt, 'w') as f:
            f.write(output)

def count_labels(input, out_txt, *labels):
    """
    Counts the number of voxels with each label in input.
    """
    arr = nib.load(input).get_data()
    out = []
    for label in labels:
        if type(label) is str:
            label = ast.literal_eval(label)
        out.append(repr((arr == label).sum()))
    output = '\n'.join(out)

    if out_txt == '-':
        print(output)
    else:
        with open(out_txt, 'w') as f:
            f.write(output)

def dice_nii(in1, in2, out_txt, *labels):
    """
    Computes Dice overlap between two nifti volumes.
    See documentation for overlap() for more info
    """
    overlap(in1, in2, out_txt, _dice, *labels)

def jaccard_nii(in1, in2, out_txt, *labels):
    """
    Computes Jaccard overlap between two nifti volumes.
    See documentation for overlap() for more info
    """
    overlap(in1, in2, out_txt, _jaccard, *labels)

