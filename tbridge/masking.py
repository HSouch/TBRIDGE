"""
All data-processing code methods for images, including profile extraction and masking procedures.
"""

from tqdm import tqdm
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from numpy import copy, ndarray, floor, nan, sum, ndarray, nanmean, nanmedian, nanstd
from photutils import detect_threshold, detect_sources, deblend_sources, make_source_mask
from photutils.aperture import CircularAnnulus, EllipticalAnnulus


def mask_cutout(cutout, config=None, nsigma=1., gauss_width=2.0, npixels=11, omit_centre=True,
                clip_negatives=True):
    """ Masks a cutout. Users can specify parameters to adjust the severity of the mask. Default
        parameters strikes a decent balance.

    Args:
        cutout: Input cutout to mask.
        config Input config file
        nsigma: The brightness requirement for objects.
        gauss_width: The width of the gaussian kernel.
        npixels: The minimum number of pixels that an object must be comprised of to be considered a source.
        omit_centre: Set as true to leave the central object unmasked.
        clip_negatives: Remove negative pixels that are 3 sigma below the median BG level
                           (This is to get rid of artifacts).
    Returns:
        The masked cutout and the associated mask data in tuple format.
    """

    # If provided with a config file, set the parameters to what is given by the user.
    # Note that the user can just use the regular specified parameters by keeping config as None
    if config is not None:
        params = config["MASK_PARAMS"]
        nsigma = params[0]
        gauss_width = params[1]
        npixels = params[2]

    mask_data = {}
    c_x, c_y = int(floor(cutout.shape[0] / 2)), int(floor(cutout.shape[1] / 2))

    # Generate background mask and statistics
    try:
        bg_mask = make_source_mask(cutout, nsigma=2, npixels=3, dilate_size=7)
    except TypeError:
        bg_mask = make_source_mask(cutout, snr=2, npixels=3, dilate_size=7)

    bg_mean, bg_median, bg_std = sigma_clipped_stats(cutout, sigma=3.0, mask=bg_mask)

    # Generate source mask
    source_mask = generate_mask(cutout, nsigma=nsigma, gauss_width=gauss_width, npixels=npixels)
    source_mask = boolean_mask(source_mask, omit=[source_mask[c_x][c_y]] if omit_centre else None)
    n_masked = sum(source_mask)

    masked_cutout = copy(cutout)
    masked_cutout[source_mask] = nan

    if clip_negatives:
        limit = bg_median - (3 * bg_std)
        masked_cutout[masked_cutout <= limit] = nan

    mask_data["BG_MEAN"] = bg_mean
    mask_data["BG_MEDIAN"] = bg_median
    mask_data["BG_STD"] = bg_std
    mask_data["N_MASKED"] = n_masked
    mask_data["P_MASKED"] = n_masked / (cutout.shape[0] * cutout.shape[1])

    return masked_cutout, mask_data


def generate_mask(cutout, nsigma=1., gauss_width=2.0, npixels=5):
    """ Generates a given mask based on the input parameters


    Args:
        cutout:     input cutout (2D-Array)
        nsigma:     Number of sigma levels above the measured background pixels are required to be
                    at to be flagged as belonging to an object.
        gauss_width: The gaussian width of the smoothing kernal for the mask.
        npixels:    The number of connected pixels required in order for something to be considered an object in the
                    segmentation map.

    Returns:
        The segment array as a 2D array
    """

    sigma = gauss_width * gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma).normalize()

    # Find threshold for cutout, and make segmentation map
    try:
        threshold = detect_threshold(cutout, nsigma=nsigma)
    except TypeError:
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


def boolean_mask(mask, omit=None):
    """ Turns a given mask (photutils segment array) into a boolean array)

    Args:
        omit: If None, mask everything. Otherwise if an int or a list, do not mask those specified.
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


def estimate_background(cutout, config, model_params=None):
    """
    Estimate the background for a cutout using some of the various available methods based on what is set in the
    configuration file.

    OPTIONS:
        ellipse: uses elliptical annuli to estimate the cutout
        circle: uses a circular annuli
        sigclip (DEFAULT): measures the background using sigma-clipping.

    """
    if config["BG_PARAMS"] == "ellipse" and model_params is not None:
        bg_mean, bg_median, bg_std = estimate_bg_elliptical_annulus(cutout,
                                                                            ellipticity=model_params["ELLIP"],
                                                                            r_50=model_params["R50"],
                                                                            pa=model_params["PA"],
                                                                            width=50,
                                                                            factor=20)
    elif config["BG_PARAMS"] == 'circle':
        bg_mean, bg_median, bg_std = estimate_bg_annulus(cutout, dynamic=True, annulus_width=50)
    else:
        bg_mean, bg_median, bg_std = estimate_background_sigclip(cutout)

    return bg_mean, bg_median, bg_std


def estimate_background_sigclip(cutout):
    """ Estimate the background mean, median, and standard deviation of a cutout using sigma-clipped-stats """
    try:
        bg_mask = make_source_mask(cutout, nsigma=2, npixels=3, dilate_size=7)
    except TypeError:
        bg_mask = make_source_mask(cutout, snr=2, npixels=3, dilate_size=7)

    bg_mean, bg_median, bg_std = sigma_clipped_stats(cutout, sigma=3.0, mask=bg_mask)

    return bg_mean, bg_median, bg_std


def estimate_bg_annulus(cutout, annulus_radius=50, annulus_width=10, dynamic=True, **kwargs):
    """
    Measure the background of a cutout using a circular annulus
    """

    args = {"nsigma": 2, "npixels": 5, "dilate_size": 11, "sigclip_iters":5}
    for arg in kwargs:
        args[arg] = kwargs[arg]

    if dynamic:
        annulus_radius = (cutout.shape[0] / 2) - annulus_width
    # First generate a background source mask for the input cutout
    bg_mask = make_source_mask(cutout,
                               nsigma=args["nsigma"],
                               npixels=args["npixels"],
                               dilate_size=args["dilate_size"],
                               sigclip_iters=args["sigclip_iters"])

    # Generate the annulus
    annulus = CircularAnnulus([cutout.shape[0] / 2, cutout.shape[1] / 2],
                              r_in=annulus_radius,
                              r_out=annulus_radius + annulus_width)

    # Generate a mask from the annulus and ensure it can be properly added to the cutout
    annulus_mask = annulus.to_mask(method='center')
    annulus_mask = annulus_mask.to_image(shape=cutout.shape)
    annulus_mask[annulus_mask == 0] = nan

    # Get the background pixels and remove any masked pixels
    bg_pixels = cutout * annulus_mask
    bg_pixels[bg_mask] = nan

    # Run our basic statistics and return them
    mean, median, std = nanmean(bg_pixels), nanmedian(bg_pixels), nanstd(bg_pixels)
    return mean, median, std


def estimate_bg_elliptical_annulus(cutout, ellipticity=0, r_50=50, pa=0, width=20, factor=10, return_mask=False):
    """
    Measure the background of a cutout using an elliptical annulus.
    """
    a_in = r_50 * factor
    b_in = a_in * (1 - ellipticity)

    a_out, b_out = a_in + width, b_in + width

    bg_mask = make_source_mask(cutout, nsigma=2, npixels=2, dilate_size=11)

    annulus = EllipticalAnnulus([cutout.shape[0] / 2, cutout.shape[1] / 2],
                                a_in=a_in, a_out=a_out,
                                b_in=b_in, b_out=b_out,
                                theta=pa)

    annulus_mask = annulus.to_mask(method='center')
    annulus_mask = annulus_mask.to_image(shape=cutout.shape)
    annulus_mask[annulus_mask == 0] = nan

    bg_pixels = cutout * annulus_mask
    bg_pixels[bg_mask] = nan

    mean, median, std = nanmean(bg_pixels), nanmedian(bg_pixels), nanstd(bg_pixels)

    if return_mask:
        return mean, median, std, bg_pixels
    else:
        return mean, median, std


def mask_cutouts(cutouts, config=None, method='standard', progress_bar=False):
    """ Mask a set of cutouts according to a certain method.

    Args:
        cutouts:
        config: User configuration file.
        method: Which method to use.
            standard : normal method. regular mask parameters. omits mask on central object
            no_central: regular mask parameters. masks central object
            background: background method: more severe mask parmeters. masks central object
        progress_bar: Use a TQDM progress bar.

    Returns:
        list of masked cutouts
    """
    masked_cutouts = []
    bg_means, bg_medians, bg_stds = [], [], []
    iterable = cutouts if not progress_bar else tqdm(cutouts)

    for cutout in iterable:
        try:
            if method == 'standard':
                masked, mask_data = mask_cutout(cutout, config=config, omit_centre=True)
            elif method == 'no_central':
                masked, mask_data = mask_cutout(cutout, config=config, omit_centre=False)
            elif method == 'background':
                masked, mask_data = mask_cutout(cutout, config=None,
                                                nsigma=0.5, gauss_width=2.0, npixels=5, omit_centre=False)
            else:
                continue

            masked_cutouts.append(masked)
            bg_means.append(mask_data["BG_MEAN"])
            bg_medians.append(mask_data["BG_MEDIAN"])
            bg_stds.append(mask_data["BG_STD"])
        except AttributeError:
            print("Cutout might be a NoneType")

    return masked_cutouts, [bg_means, bg_medians, bg_stds]


def estimate_background_set(cutouts):
    """
    Estimates the background values for a set of cutouts.
    :param cutouts:
    :return:
    """
    bg_means, bg_medians, bg_stds = [], [], []

    for cutout in cutouts:
        bg_mean, bg_median, bg_std = estimate_background_sigclip(cutout)

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
