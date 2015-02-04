from __future__ import print_function
from __future__ import division
import sys
import ast

import numpy as np
from numpy import newaxis as nax
import nibabel as nib

def copy_header(infile, headerfile, outfile):
    """
    Takes the header/affine info from headerfile and the image from infile,
    combines them, and saves the result to outfile.

    Inputs
    ------
    infile : filename from which to extract the data
    headerfile : filename from which to extract the header & affine matrix
    outfile : a nifti filename to save the result
    """
    inp = nib.load(infile)
    header = nib.load(headerfile)

    out = nib.Nifti1Image(inp.get_data(), header=header.get_header(), affine=header.get_affine())

    out.to_filename(outfile)

def neutralize_affine(infile, outfile):
    """
    Sets the affine matrix to be diagonal, ignoring rotations.  Be careful:
    this can erase orientation information in your data, and should only be
    used with data that either:
        a) you know is already nearly axis-aligned, or
        b) has been conformed to freesurfer standard orientation (e.g., with
           mri_convert)
    """
    nii = nib.load(infile)
    data = nii.get_data()
    affine = nii.get_affine()[:3, :3]
    header = nii.get_header()

    freesurfer_orientation = np.array([[-1,0,0],[0,0,-1],[0,1,0]])

    if np.allclose(freesurfer_orientation, np.sign(affine)):
        data = data.transpose(0,2,1)[-1::-1, :, -1::-1]
    else:
        data = nii.get_data()

    pixdim = header['pixdim'][1:4]
    good_affine = np.eye(4)
    np.fill_diagonal(good_affine, np.hstack((pixdim, [0])))

    # Make the image start at (0,0,0)
    #good_affine[:3,3] = -(header['dim'][1:4] * pixdim)/2
    good_affine[:3,3] = 1
    new = nib.Nifti1Image(data,header=nii.get_header(),affine=good_affine)
    new.to_filename(outfile)


def merge_warps(dimensions, infile_pattern, original_file, outfile):
    """
    Splits a 3D warp along its last dimension
    """
    dimensions = int(dimensions)
    header_source = nib.load(original_file)

    data_arrays = []
    for d in xrange(int(dimensions)):
        nii = nib.load(infile_pattern % d)
        dat = nii.get_data()
        if dimensions == 2:
            dat = dat[...,np.newaxis,np.newaxis, np.newaxis]
        elif dimensions == 3:
            dat = dat[...,np.newaxis,np.newaxis]
        data_arrays.append(dat)
    data = np.concatenate(data_arrays, 4)
    out = nib.Nifti1Image(data, header=header_source.get_header(), affine=nii.get_affine())
    out.to_filename(outfile)



def split_warp(dimensions, infile, outfile_pattern):
    """
    Splits a 3D warp along its last dimension
    """
    dimensions = int(dimensions)
    nii = nib.load(infile)
    dat = nii.get_data()
    for d in xrange(int(dimensions)):
        h = nii.get_header().copy()
        smallnii = nib.Nifti1Image(dat[...,0,d], affine=nii.get_affine(), header=h)
        smallnii.update_header()
        smallnii.to_filename(outfile_pattern % d)

def convert_type(infile, outfile, newtype, normalization='none'):
    """
    Converts a nifti file from one type to another.

    Inputs
    ------
    infile: a filename from which to read data
    outfile: a filename to which to save data
    newtype: the new datatype
    normalization: 'none', no renormalization
                   these are only meant for use with unsigned types:
                   'data', renormalizes based on min/max of data
                   'prob', renormalizes assuming data is between 0,1
                   numeric value, renormalizes assuming 0 to this val
    """
    nii = nib.load(infile)
    data = nii.get_data()
    header = nii.get_header()
    header.set_data_dtype(newtype)
    newmax = np.iinfo(newtype).max
    if normalization == 'none':
        pass
    elif normalization == 'data':
        (min, max) = (data.min(), data.max())
        data = (data - min) / (max-min) * newmax
    elif normalization == 'prob':
        data = data * newmax
    else:
        try:
            maxval = float(normalization)
        except ValueError:
            raise
        data = data / maxval * newmax

    data = data.astype(newtype)
    new = nib.Nifti1Image(data, header=header, affine=nii.get_affine())
    new.to_filename(outfile)

