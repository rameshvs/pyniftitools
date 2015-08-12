from __future__ import print_function
from __future__ import division

import ast

import numpy as np
import nibabel as nib

def warp_ssd(in1, in2, template, output):
    """
    Computes the SSD between two warps, using template for header info.
    """
    ssd = np.sum((nib.load(in1).get_data() - nib.load(in2).get_data())**2, 4).squeeze()
    template_nii = nib.load(template)
    out = nib.Nifti1Image(ssd, header=template_nii.get_header(), affine=template_nii.get_affine())
    out.to_filename(output)

def _masked_threshold_arr(in_arr, threshold, label_arr, exclude_arr, direction, *labels):

    labels = map(int, labels)
    if len(labels) == 0:
        labels = [1]
    mask_arr = np.zeros(label_arr.shape)
    for label in labels:
        mask_arr = np.logical_or(mask_arr, label_arr == label)
    mask_arr = np.logical_and(mask_arr, np.logical_not(exclude_arr))
    if direction == 'greater':
        binary = in_arr > threshold
    elif direction == 'less':
        binary = in_arr < threshold
    if len(binary.shape) == 4:
        binary = binary[:,:,:,0]
    return np.logical_and(mask_arr, binary)

def ssd(in1, in2, output):
    first = nib.load(in1).get_data() 
    second = nib.load(in2).get_data()
    if first.shape == (256,256,256) and second.shape == (170,170,170):
        ssd = np.sum((first[43:-43,43:-43,43:-43]- second)**2)
    else:
        ssd = np.sum((first- second)**2)
    with open(output, 'w') as f:
        f.write(repr(ssd))

def gaussian_blur(infile, outfile, sigma):
    from scipy.ndimage.filters import gaussian_filter
    if type(sigma) is str:
        sigma = ast.literal_eval(sigma)
    in_nii = nib.load(infile)

    in_arr = in_nii.get_data()
    out_arr = np.zeros_like(in_arr)
    if len(in_arr.shape) == 5:
        assert in_arr.shape[3] == 1
        assert in_arr.shape[4] == 3
        # Warp: blur x,y,z separately
        for i in xrange(3):
            gaussian_filter(
                    in_arr[:,:,:,0,i],
                    sigma=sigma,
                    output=out_arr[:,:,:,0,i])
    elif len(in_arr.shape) == 3:
            gaussian_filter(
                    in_arr[:,:,:],
                    sigma=sigma,
                    output=out_arr[:,:,:])

    out_nii = nib.Nifti1Image(out_arr, header=in_nii.get_header(), affine=in_nii.get_affine())
    out_nii.to_filename(outfile)




def _masked_threshold(infile, threshold, outfile, mode='scalar', excludefile='-',
        labelfile='-', direction='greater', units='mm', *labels):
    """
    Counts how many voxels are above/below a threshold.

    Inputs
    ------
    infile : filename with the input image
    threshold : numeric threshold
    outfile : a file to write to. If mode is 'scalar', writes a number.
                  Otherwise, writes a nifti volume
    mode : 'scalar' (writes a single number with the total volume)
            or 'nii' (writes a binary mask with the result of thresholding)
    excludefile: filename of a binary mask to exclude from the results.
                 use '-' for no mask
    labelfile : filename of a mask/labelmap to count within. use '-' for no mask
    direction : 'greater' or 'less'. 'greater' counts intensities
                above threshold, and 'less' counts intensities
                below.
    units : either 'mm' (writes results in mm^3) or 'voxels'
    """
    if type(threshold) is str:
        threshold = ast.literal_eval(threshold)
    nii = nib.load(infile)
    data = nii.get_data()

    if labelfile == '-':
        label_arr = np.ones(data.shape)
    else:
        label_arr = nib.load(labelfile).get_data()

    if excludefile == '-':
        exclude_arr = np.zeros(data.shape)
    else:
        exclude_arr = nib.load(excludefile).get_data()
    vol = _masked_threshold_arr(data, threshold, label_arr, exclude_arr, direction, *labels)
    if mode == 'scalar':
        count = vol.sum()
        if units == 'mm':
            count *= nii.get_header()['pixdim'][1:4].prod()
        elif units != 'voxels':
            raise ValueError("I expected units in mm or voxels, but you asked for  " + units)

        with open(outfile, 'w') as f:
            f.write(str(count))
    else:
        out = nib.Nifti1Image(vol, header=nii.get_header(),
                affine=nii.get_affine())
        out.to_filename(outfile)

def masked_threshold(infile, threshold, outfile, excludefile='-',
        labelfile='-', direction='greater', *labels):
    """
    Thresholds a volume within a mask, and saves a new binary
    mask of voxels that are above/below the threshold.

    Inputs
    ------
    infile : filename with the input image
    threshold : numeric threshold
    outfile : a file to write the output image to
    labelfile : filename of a mask/labelmap to count within. use '-' for no mask
    direction : 'greater' or 'less'. 'greater' counts intensities
                above threshold, and 'less' counts intensities below.
    labels : labels within which to do thresholding
    """
    _masked_threshold(infile, threshold, outfile, 'nii', excludefile,
            labelfile, direction, 'mm', *labels)

def masked_threshold_count(infile, threshold, outfile, excludefile='-',
        labelfile='-', direction='greater', units='mm', *labels):
    """
    Thresholds a volume within a mask, and saves how much
    of the volume is above/below the threshold.

    Inputs
    ------
    infile : filename with the input image
    threshold : numeric threshold
    outfile : a text file to write the output value to
    labelfile : filename of a mask/labelmap to count within. use '-' for no mask
    direction : 'greater' or 'less'. 'greater' counts intensities
                above threshold, and 'less' counts intensities below.
    units : either 'mm' (writes results in mm^3) or 'voxels'
    labels : labels within which to do thresholding
    """
    _masked_threshold(infile, threshold, outfile, 'scalar', excludefile,
            labelfile, direction, units, *labels)

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

def pad(niiFileName, paddedNiiFileName, maskNiiFileName, padAmountMM='30'):
    """
    pad a nifti and save the nifti and the relevant mask.

    Example arguments:
    niiFileName = '10529_t1.nii.gz'
    paddedNiiFileName = 'padded_10529_t1.nii.gz'
    maskNiiFileName = 'padded_10529_t1_mask.nii.gz'
    padAmountMM = '30'; [default]
    """

    # padding amount in mm
    padAmountMM = int(padAmountMM)

    # load the nifti
    nii = nib.load(niiFileName)

    # get the amount of padding in voxels
    pixdim = nii.get_header()['pixdim'][1:4]
    padAmount = np.ceil(padAmountMM / pixdim)
    dims = nii.get_header()['dim'][1:4]
    assert np.all(dims.shape == padAmount.shape)
    newDims = dims + padAmount * 2

    # compute where the center is for padding
    center = newDims/2
    starts = np.round(center - dims/2)
    ends = starts + dims

    # compute a slice object with the start/end of the center subvolume
    slicer = [slice(start, end) for (start, end) in zip(starts, ends)]

    # set the subvolume in the center of the image w/the padding around it
    vol = np.zeros(newDims)
    vol[slicer] = nii.get_data()
    volMask = np.zeros(newDims)
    volMask[slicer] = np.ones(dims)

    # update affine
    affine = nii.get_affine()
    affine[:3, 3] -= padAmountMM
    # create niftis
    newNii = nib.Nifti1Image(vol, header=nii.get_header(), affine=affine)
    newNiiMask = nib.Nifti1Image(volMask, header=nii.get_header(), affine=affine)

    # save niftis
    newNii.to_filename(paddedNiiFileName)
    newNiiMask.to_filename(maskNiiFileName)
    return (newNii, newNiiMask)

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

