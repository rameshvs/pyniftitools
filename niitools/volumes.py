from __future__ import print_function
from __future__ import division

import numpy as np
from numpy import newaxis as nax
import nibabel as nib

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

    trim_slicing('in.nii.gz', 'out.nii.gz', [slice(30,200), slice(20, 280, 2), None])

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

