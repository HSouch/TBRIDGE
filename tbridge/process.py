"""
All data-processing code methods for images, including profile extraction and masking procedures.
"""

from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from numpy import median, mean, max, copy, pi
from numpy import ndarray, log, exp, sqrt
from numpy import unravel_index, argmax, floor, ceil
from photutils import detect_threshold, detect_sources, deblend_sources, data_properties
from photutils.isophote import Ellipse, EllipseGeometry
from scipy.special import gamma


def mask_cutout(cutout, nsigma=1., gauss_width=2.0, npixels=5):
    """ Masks a cutout using segmentation and deblending using watershed"""
    mask_data = {}

    # Generate a copy of the cutout just to prevent any weirdness with numpy pointers
    cutout_copy = copy(cutout)

    sigma = gauss_width * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma)
    kernel.normalize()

    # Find threshold for cutout, and make segmentation map
    threshold = detect_threshold(cutout, snr=nsigma)
    segments = detect_sources(cutout, threshold, npixels=npixels,
                              filter_kernel=kernel)

    # Attempt to de-blend. Return original segments upon failure.
    try:
        deb_segments = deblend_sources(cutout, segments, npixels=5,
                                       filter_kernel=kernel)
    except ImportError:
        print("Skimage not working!")
        deb_segments = segments
    except:
        # Don't do anything if it doesn't work
        deb_segments = segments

    segment_array = deb_segments.data

    # Center pixel values. (Assume that the central segment is the image, which is should be)
    c_x, c_y = floor(segment_array.shape[0] / 2), floor(segment_array.shape[1] / 2)
    central = segment_array[int(c_x)][int(c_y)]

    # Estimate Background, and min/max values
    bg_total, bg_pixels, bg_pixel_array = 0, 0, []
    min_val, max_val, = cutout_copy[0][0], cutout_copy[0][0]
    for x in range(0, segment_array.shape[0]):
        for y in range(0, segment_array.shape[1]):
            if segment_array[x][y] == 0:
                bg_total += cutout_copy[x][y]
                bg_pixels += 1
                bg_pixel_array.append(cutout_copy[x][y])

    bg_estimate = bg_total / bg_pixels
    mask_data["BG_EST"] = bg_estimate
    mask_data["BG_MED"] = median(bg_pixel_array)
    mask_data["N_OBJS"] = segments.nlabels
    mask_data["MIN_VAL"] = min_val
    mask_data["MAX_VAL"] = max_val

    # Return input image if no need to mask
    if segments.nlabels == 1:
        mask_data["N_MASKED"] = 0
        return cutout_copy, mask_data

    num_masked = 0
    # Mask pixels
    for x in range(0, segment_array.shape[0]):
        for y in range(0, segment_array.shape[1]):
            if segment_array[x][y] not in (0, central):
                cutout_copy[x][y] = bg_estimate
                num_masked += 1
    mask_data["N_MASKED"] = num_masked

    return cutout_copy, mask_data


def estimate_background(cutout):
    """ Simple background detecting using super-pixel method"""
    x_step = int(cutout.shape[0] / 10)
    y_step = int(cutout.shape[1] / 10)

    super_pixel_medians, super_pixel_rms_vals = [], []

    for x in range(0, cutout.shape[0] - x_step, x_step):
        for y in range(0, cutout.shape[1] - y_step, y_step):
            super_pixel = cutout[y: y + y_step, x: x + x_step]

            super_pixel_contents = []
            for m in range(0, super_pixel.shape[0]):
                for n in range(0, super_pixel.shape[1]):
                    super_pixel_contents.append(super_pixel[m][n])

            super_pixel_medians.append(median(super_pixel_contents))
            super_pixel_rms_vals.append(sqrt((mean(super_pixel_contents) - median(super_pixel_contents)) ** 2))

    return median(super_pixel_medians), median(super_pixel_rms_vals)


def b(n):
    """ Get the b_n normalization constant for the sersic profile. From Graham and Driver."""
    return 2 * n - (1 / 3) + (4 / (405 * n)) + (46 / (25515 * (n ** 2)))


def i_at_r50(mag, n=2, r_50=2, m_0=27):
    """ Get the intensity at the half-light radius """
    b_n = b(n)
    l_tot = 10 ** ((mag - m_0) / -2.5)
    denom = (r_50 ** 2) * 2 * pi * n * exp(b_n) * gamma(2 * n) / (b_n ** (2 * n))
    i_e = l_tot / denom

    return i_e



