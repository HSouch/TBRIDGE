"""
All data-processing code methods for images, including profile extraction and masking procedures.
"""

from tqdm import tqdm
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from numpy import copy, ndarray, floor
from photutils import detect_threshold, detect_sources, deblend_sources


def mask_cutout(cutout, nsigma=1., gauss_width=2.0, npixels=5, omit_centre=True):
    """
    Masks a cutout. Users can specify parameters to adjust the severity of the mask. Default
    parameters striks a decent balance.
    :param cutout: Input cutout to mask.
    :param nsigma: The brightness requirement for objects.
    :param gauss_width: The width of the gaussian kernel.
    :param npixels: The minimum number of pixels that an object must be comprised of to be considered a source.
    :param omit_centre: Set as true to leave the central object unmasked.
    :return:
    """
    mask_data = {}
    c_x, c_y = int(floor(cutout.shape[0] / 2)), int(floor(cutout.shape[1] / 2))

    # Generate background mask and statistics
    bg_mask = generate_mask(cutout, nsigma=0.5, gauss_width=2.0, npixels=5)
    bg_mask = boolean_mask(bg_mask)
    bg_mean, bg_median, bg_std = sigma_clipped_stats(cutout, sigma=3.0, mask=bg_mask)

    # Generate source mask
    source_mask = generate_mask(cutout, nsigma=nsigma, gauss_width=gauss_width, npixels=npixels)
    source_mask = boolean_mask(source_mask, omit=[source_mask[c_x][c_y]] if omit_centre else None)

    n_masked = sum(source_mask)

    masked_cutout = copy(cutout)
    masked_cutout[source_mask] = bg_median

    mask_data["BG_MEAN"] = bg_mean
    mask_data["BG_MEDIAN"] = bg_median
    mask_data["BG_STD"] = bg_std
    mask_data["N_MASKED"] = n_masked
    mask_data["P_MASKED"] = n_masked / (cutout.shape[0] * cutout.shape[1])

    return masked_cutout, mask_data


def generate_mask(cutout, nsigma=1., gauss_width=2.0, npixels=5):
    """ Gemerates a given mask based on the input parameters """

    sigma = gauss_width * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma)
    kernel.normalize()

    # Find threshold for cutout, and make segmentation map
    threshold = detect_threshold(cutout, snr=nsigma)
    segments = detect_sources(cutout, threshold, npixels=npixels, filter_kernel=kernel)

    # Attempt to de-blend. Return original segments upon failure.
    try:
        deb_segments = deblend_sources(cutout, segments, npixels=npixels, filter_kernel=kernel)
    except ImportError:
        print("Skimage not working!")
        deb_segments = segments
    except:
        # Don't do anything if it doesn't work
        deb_segments = segments

    segment_array = deb_segments.data

    return segment_array


def boolean_mask(mask, omit: list = None):
    """
    Turns a given mask (photutils segment array) into a boolean array)
    :param mask:
    :param omit:
    :return:
    """
    if omit is None:
        omit = []
    elif type(omit) == int:
        omit = [omit]

    bool_mask = ndarray(mask.shape, dtype="bool")

    bool_mask[:] = False
    bool_mask[mask > 0] = True
    for val in omit:
        bool_mask[mask == val] = False

    return bool_mask


def estimate_background(cutout):
    """ Simple background detecting using super-pixel method"""
    bg_mask = generate_mask(cutout, nsigma=0.5, gauss_width=2.0, npixels=5)
    bg_mask = boolean_mask(bg_mask)
    bg_mean, bg_median, bg_std = sigma_clipped_stats(cutout, sigma=3.0, mask=bg_mask)

    return bg_mean, bg_median, bg_std


def mask_cutouts(cutouts, method='standard', progress_bar=False):
    """
    Mask a set of cutouts according to a certain method.
    :param cutouts:
    :param method: Which method to use.
        standard : normal method. regular mask parameters. omits mask on central object
        no_central: regular mask parameters. masks central object
        background: background method: more severe mask parmeters. masks central object
    :return: list of masked cutouts
    """
    masked_cutouts = []

    iterable = cutouts if not progress_bar else tqdm(cutouts)

    for cutout in iterable:
        if method == 'standard':
            masked, mask_data = mask_cutout(cutout, omit_centre=True)
        elif method == 'no_central':
            masked, mask_data = mask_cutout(cutout, omit_centre=False)
        elif method == 'background':
            masked, mask_data = mask_cutout(cutout, nsigma=0.5, gauss_width=2.0, npixels=5, omit_centre=False)
        else:
            continue

        masked_cutouts.append(masked)

    return masked_cutouts


def estimate_background_set(cutouts):
    """
    Estimates the background values for a set of cutouts.
    :param cutouts:
    :return:
    """
    bg_means, bg_medians, bg_stds = [], [], []

    for cutout in cutouts:
        bg_mean, bg_median, bg_std = estimate_background(cutout)

        bg_means.append(bg_mean)
        bg_medians.append(bg_median)
        bg_stds.append(bg_std)

    return bg_means, bg_medians, bg_stds


def subtract_backgrounds(profile_set, background_array):
    """
    Generate an array of tables identical to the input except the respective backgrounds
    are subtracted from the intensity array for each table.
    :param profile_set: Set of profiles (in the photutiuls isolist format)
    :param background_array: Array of background values to subtract from each profile table.
    :return: List of profiles of length len(profile_set)
    """
    bg_subtracted_tables = []

    for i in range(0, len(profile_set)):
        this_table = profile_set[i]
        isotable_localsub = Table()

        for col in this_table.colnames:
            isotable_localsub[col] = copy(this_table[col])
        isotable_localsub["intens"] = ((isotable_localsub["intens"]) - background_array[i])

        bg_subtracted_tables.append(isotable_localsub)

    return bg_subtracted_tables
