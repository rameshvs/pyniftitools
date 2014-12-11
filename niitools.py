#!/usr/bin/env python
"""
This file contains useful tools for dealing with NifTI files and performing
common operations. Input and output files will typically be in nifti (.nii
or .nii.gz) format.

Can be used from the command line.
"""
from __future__ import print_function
from __future__ import division
import sys
import ast

import numpy as np
import nibabel as nib

class SliceMaker(object):
    def __getitem__(self, slic):
        return slic

slicer = SliceMaker

def mask(infile, maskfile, outfile):
    """
    Given a binary mask (maskfile) and an input image (infile),
    sets all areas outside the mask (i.e., where the mask image is 0)
    to be 0, and saves the result to outfile.

    Inputs
    ------
    infile : filename with the input image
    maskfile : filename with the binary mask image
    outfile : filename to save the result. All voxels in the input where
              the mask=0 are set to 0, and the rest are left as they are.
    """
    inp = nib.load(infile)
    mask = nib.load(maskfile)
    binary_mask = (mask.get_data() == 0)

    masked = inp.get_data()
    masked[binary_mask] = 0

    out = nib.Nifti1Image(masked, header=inp.get_header(), affine=inp.get_affine())

    out.to_filename(outfile)

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
    good_affine[:3,3] = -(header['dim'][1:4] * pixdim)/2
    new = nib.Nifti1Image(data,header=nii.get_header(),affine=good_affine)
    new.to_filename(outfile)


def trim_bounding_box(infile, outfile, xmin, ymin, zmin, xmax, ymax, zmax):
    """
    Trims the input volume using the specified bounding box.
    """
    slicers = [slice(xmin, xmax), slice(ymin, ymax), slice(zmin, zmax)]
    trim_slicing(infile, outfile, slicers, update_headers=True)

def trim_slicing(infile, outfile, slices=None, update_headers=True):
    """
    Trims the input volume using the list of slice objects provided and
    saves the result.

    Inputs
    ------
    infile : a filename from which to read data
    outfile : a filename to which to save data
    slices : a list of slice objects. Defaults to values that usually work
             well for 3D 1mm-resolution brain MRI data.

    SEE ALSO: slicer, SliceMaker.

    Examples
    --------
    The following two examples are equivalent. Both trim the x
    dimension and y dimension and downsample the y dimension,
    while leaving the z dimension untouched.

    trim_slicing('in.nii.gz', 'out.nii.gz', [slice(30,200), slice(20, 280, 2), None]

    trim_slicing('in.nii.gz', 'out.nii.gz', niitools.slicer[30:200, 20:280,2, :])
    """
    if slices is None:
        slices = [slice(49,211), slice(22,220), slice(38,183)]
    inp = nib.load(infile)

    aff = inp.get_affine()
    if update_headers:
        # modify origin according to affine * what we added
        aff[:3, -1] += np.dot(aff[:3,:3], np.array([s.start for s in slices]))
    out = nib.Nifti1Image(inp.get_data()[slices], header=inp.get_header(), affine=aff)

    out.to_filename(outfile)

def crop_to_bounding_box(infile, outfile):
    """
    Crops the volume in infile to locations where it is nonzero.
    Prints the resulting bounding box in the following format:
    xmin ymin zmin xmax ymax zmax
    """
    nii = nib.load(infile)
    aff = nii.get_affine()
    data = nii.get_data()
    minmaxes = []
    slicing = []
    for axis, otheraxes in enumerate([[2,1], [2,0], [1,0]]):
        one_axis = np.apply_over_axes(np.sum, data, otheraxes).squeeze()
        # hack because image has a weird bright patch
        (nonzero_locs,) = np.where(one_axis)
        minmaxes.append((nonzero_locs.min(), nonzero_locs.max()))


    minima = [int(min) for (min, max) in minmaxes]
    maxima = [int(max) for (min, max) in minmaxes]
    slicing = [slice(min, max, None) for (min, max) in minmaxes]
    aff[:3, -1] += minima
    out = nib.Nifti1Image(data[slicing], header=nii.get_header(), affine=aff)
    out.update_header()
    out.to_filename(outfile)
    print(" ".join(map(str,minima)))
    print(" ".join(map(str,maxima)))

if __name__ == '__main__':
    help = \
"""USAGE: {fname} [mask | copy_header | neutralize_affine | trim ] ...

mask
    {fname} mask <infile> <maskfile> <outfile>
    masks data in infile using binary mask: data outside the mask is set to 0

copy_header
    {fname} copy_header <infile> <headerfile> <outfile>
    copies data from infile & header from headerfile, saves result to outfile

neutralize_affine
    {fname} neutralize_affine <infile> <outfile>
    removes affine information from header of infile, saves result to outfile

trim
    {fname} trim <infile> <outfile> [<xmin <ymin> <zmin> <xmax> <ymax> <zmax>]
    trims the volume in infile; if bbox not given, assumes 256^3 brain MR

split
    {fname} split <dimension> <inwarp> <splitpattern>
    Splits a warp into separate volumes/images for each dimension.

merge
    {fname} merge <dimension> <inpattern> <templatewarp> <outputfile>
    Splits a warp into separate volumes/images for each dimension.

upsample
    {fname} upsample <input> <output> <output_mask> <axis> <ratio>
    Upsamples an image along the given axis by the given amount.
""".format(fname=sys.argv[0])

    if len(sys.argv) < 3:
        print(help, file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == 'mask':
        if len(args) == 3:
            mask(*args)
    elif cmd == 'copy_header':
        if len(args) == 3:
            copy_header(*args)
    elif cmd == 'neutralize_affine':
        if len(args) == 2:
            neutralize_affine(*args)
    elif cmd == 'trim':
        if len(args) == 2:
            trim_slicing(*args)
        elif len(args) == 8:
            trim_bounding_box(args[0], args[1], *map(int, args[2:]))
    else:
        print(help, file=sys.stderr)
        sys.exit(1)

