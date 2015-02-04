from __future__ import print_function
from __future__ import division
import sys
import ast

import numpy as np
from numpy import newaxis as nax
import nibabel as nib

def downsample_axis(infile, outfile, axis, new_pixdim, method='linear'):
    """
    Downsamples a volume along a specified axis.

    Inputs
    ------
    infile : a filename from which to read data
    outfile : a filename to which to save data
    axis : the axis along which to downsample
    pixdim_ratio : the ratio by which to decrease pixdim.
    method : interpolation method ('linear' or 'nearest')
    """
    if type(new_pixdim) is str:
        new_pixdim = ast.literal_eval(new_pixdim)
    if type(axis) is str:
        axis = ast.literal_eval(axis)
    from scipy.interpolate import interpn
    nii = nib.load(infile)
    hdr = nii.get_header()
    aff = nii.get_affine()
    data = nii.get_data().astype('float32')

    in_coords = []
    out_coords = []
    affine_modifier = np.eye(3)
    for ax in [0,1,2]:
        in_coords.append(np.arange(256))
        if ax == axis:
            out_slice = slice(0, 252, new_pixdim)
            affine_modifier[ax,ax] = new_pixdim
        else:
            out_slice = slice(0, 256)
        out_coords.append(out_slice)

    out_grid = np.mgrid[out_coords].transpose(1,2,3,0)

    new_data = interpn(in_coords, data, out_grid, method=method, fill_value=None)
    hdr['pixdim'][1+axis] = new_pixdim
    # Multiply affine matrix by resampling matrix.
    # WARNING: no guarantees this'll work for non-axis-aligned images...
    aff[:3, :3] = np.dot(affine_modifier, aff[:3, :3])
    #new_aff = np.vstack((np.dot(affine_modifier, aff[:-1,:]), aff[-1:,:]))

    out = nib.Nifti1Image(new_data.astype('uint8'), header=hdr.copy(), affine=aff)
    out.update_header()
    out.to_filename(outfile)

def upsample_axis(infile, outfile, outmask, axis, pixdim_ratio, method='linear'):
    """
    Upsamples a volume along a specified axis.

    Inputs
    ------
    infile : a filename from which to read data
    outfile : a filename to which to save data
    outmask : a filename to which to save the upsampling mask
    axis : the axis along which to upsample
    pixdim_ratio : the ratio by which to increase pixdim.
                   if integer, inserts (#-1) slices between each
    method : interpolation method ('linear' or 'nearest')
    """
    if type(pixdim_ratio) is str:
        pixdim_ratio = ast.literal_eval(pixdim_ratio)
    if type(axis) is str:
        axis = ast.literal_eval(axis)
    from scipy.interpolate import interpn
    nii = nib.load(infile)
    hdr = nii.get_header()
    aff = nii.get_affine()
    data = nii.get_data().astype('float32')

    in_coords = []
    out_coords = []
    affine_modifier = np.eye(3)
    mask_coords = []
    for ax in [0,1,2]:
        pixdim = hdr['pixdim'][1+ax]
        # cheat/hack
        if abs(pixdim - round(pixdim)) < .05:
            pixdim = round(pixdim)
        n_slices = data.shape[ax]

        cap = pixdim * (n_slices-1) + (.01*pixdim)

        slicer = np.arange(0, cap, pixdim)
        in_coords.append(slicer)
        if ax == axis:
            out_slice = slice(0, cap, pixdim / pixdim_ratio)
            affine_modifier[ax,ax] = 1/pixdim_ratio
            mask_slice = slice(0, n_slices*pixdim_ratio, pixdim_ratio)
        else:
            out_slice = slice(0, cap, pixdim)
            mask_slice = slice(0, n_slices)
        out_coords.append(out_slice)
        mask_coords.append(mask_slice)

    out_grid = np.mgrid[out_coords].transpose(1,2,3,0)
    slices = [slice(None) for i in [0,1,2]] + [axis]
    slices[axis] = -1
    assert np.allclose(out_grid[slices], out_grid[slices].max())

    # Hack to avoid numerical issues where the output coordinates
    # might be very slightly beyond the range due to floating point
    # error
    if np.allclose(out_grid[slices].max(), in_coords[axis][-1]):
        out_grid[slices] = in_coords[axis][-1]

    new_data = interpn(in_coords, data, out_grid, method=method, fill_value=None)
    mask = np.zeros_like(new_data)
    mask[mask_coords] = 1
    hdr['pixdim'][1+axis] = pixdim / pixdim_ratio
    # Multiply affine matrix by resampling matrix.
    # WARNING: no guarantees this'll work for non-axis-aligned images...
    aff[:3, :3] = np.dot(affine_modifier, aff[:3, :3])
    #new_aff = np.vstack((np.dot(affine_modifier, aff[:-1,:]), aff[-1:,:]))

    out = nib.Nifti1Image(new_data, header=hdr.copy(), affine=aff)
    out.update_header()
    out.to_filename(outfile)

    out_mask = nib.Nifti1Image(mask, header=hdr.copy(), affine=aff)
    out_mask.update_header()
    out_mask.to_filename(outmask)
