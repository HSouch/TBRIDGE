from numpy import max, pi, log
from numpy import unravel_index, argmax, ceil
from photutils import data_properties

# from isophote_l import Ellipse, EllipseGeometry
from tbridge.isophote_l import Ellipse, EllipseGeometry

from tqdm import tqdm

import sys


def isophote_fitting(data, config=None, centre_method='standard', fit_method='standard', maxrit=None):
    """
    Generates a table of results from isophote fitting analysis. This uses photutils Isophote procedure, which is
    effectively IRAF's Ellipse() method.
    Iterates over many possible input ellipses to force a higher success rate.
    :return:  The table of results, or an empty list if not fitted successfully.
    """
    # Set-up failsafe in case of strange infinte loops in photutils
    # warnings.filterwarnings("error")

    fail_count, max_fails = 0, 1000
    linear = False if config is None else config["LINEAR"]
    step = 1. if config is None else config["LINEAR_STEP"]
    verbose = False if config is None else config["VERBOSE"]
    test_verbose = False if config is None else config["TEST_VERBOSE"]

    # Get centre of image and cutout halfwidth
    if centre_method == 'standard':
        centre = (data.shape[0]/2, data.shape[1]/2)
    elif centre_method == 'max':
        centre = unravel_index(argmax(data), data.shape)
    else:
        centre = (data.shape[0] / 2, data.shape[1] / 2)

    cutout_halfwidth = max((ceil(data.shape[0] / 2), ceil(data.shape[1] / 2)))

    fitting_list = []

    # First, try obtaining morphological properties from the data and fit using that starting ellipse
    try:
        morph_cat = data_properties(log(data))
        r = 2.0
        pos = (morph_cat.xcentroid.value, morph_cat.ycentroid.value)

        a = morph_cat.semimajor_axis_sigma.value * r
        b = morph_cat.semiminor_axis_sigma.value * r
        theta = morph_cat.orientation.value

        geometry = EllipseGeometry(pos[0], pos[1], sma=a, eps=(1 - (b / a)), pa=theta)
        flux = Ellipse(data, geometry)
        fitting_list = flux.fit_image(maxit=100, maxsma=cutout_halfwidth, step=step, linear=linear,
                                      maxrit=cutout_halfwidth / 3)
        if len(fitting_list) > 0:
            return fitting_list

    except KeyboardInterrupt:
        sys.exit(1)
    except (RuntimeError, ValueError, OverflowError, IndexError):
        fail_count += 1
        if fail_count >= max_fails:
            return []

    # If that fails, test a parameter space of starting ellipses
    try:
        for angle in range(0, 180, 45):
            for sma in range(2, 26, 5):
                for eps in (0.3, 0.5, 0.9):
                    geometry = EllipseGeometry(float(centre[0]), float(centre[1]), eps=eps,
                                               sma=sma, pa=angle * pi / 180.)
                    flux = Ellipse(data, geometry)
                    fitting_list = flux.fit_image(maxsma=cutout_halfwidth, step=step, linear=linear,
                                                  maxrit=cutout_halfwidth / 3)
                    if len(fitting_list) > 0:
                        return fitting_list

    except KeyboardInterrupt:
        sys.exit(1)
    except (RuntimeError, ValueError, OverflowError, IndexError):

        # print("RuntimeError or ValueError")
        fail_count += 1
        if fail_count >= max_fails:
            return []
    except IndexError:
        fail_count += 1
        if fail_count >= max_fails:
            return []

    return fitting_list


def _isophote_fitting_tester(data, config=None, centre_method='standard', fit_method='standard', maxrit=None):
    """
    Identical to tbridge.isophote_fitting() but with far fewer exception handling cases to check for issues in
    isophote fitting (the internal code from Photutils).
    """
    # Set-up failsafe in case of strange infinte loops in photutils
    # warnings.filterwarnings("error")

    fail_count, max_fails = 0, 1000
    linear = False if config is None else config["LINEAR"]
    step = 1. if config is None else config["LINEAR_STEP"]

    # Get centre of image and cutout halfwidth
    if centre_method == 'standard':
        centre = (data.shape[0] / 2, data.shape[1] / 2)
    elif centre_method == 'max':
        centre = unravel_index(argmax(data), data.shape)
    else:
        centre = (data.shape[0] / 2, data.shape[1] / 2)

    cutout_halfwidth = max((ceil(data.shape[0] / 2), ceil(data.shape[1] / 2)))

    fitting_list = []

    # First, try obtaining morphological properties from the data and fit using that starting ellipse
    try:
        morph_cat = data_properties(log(data))
        r = 2.0
        pos = (morph_cat.xcentroid.value, morph_cat.ycentroid.value)

        a, b = morph_cat.semimajor_axis_sigma.value * r, morph_cat.semiminor_axis_sigma.value * r
        theta = morph_cat.orientation.value

        geometry = EllipseGeometry(pos[0], pos[1], sma=a, eps=(1 - (b / a)), pa=theta)
        flux = Ellipse(data, geometry)
        fitting_list = flux.fit_image(maxit=100, maxsma=cutout_halfwidth, step=step, linear=linear,
                                      maxrit=cutout_halfwidth / 3)
        if len(fitting_list) > 0:
            return fitting_list

    except KeyboardInterrupt:
        sys.exit(1)
    except OverflowError:
        fail_count += 1
        if fail_count >= max_fails:
            return []

    # If that fails, test a parameter space of starting ellipses
    try:
        for angle in range(0, 180, 45):
            for sma in range(4, 26, 5):
                for eps in (0.3, 0.5, 0.9):
                    geometry = EllipseGeometry(float(centre[0]), float(centre[1]), eps=eps,
                                               sma=sma, pa=angle * pi / 180.)
                    flux = Ellipse(data, geometry)
                    fitting_list = flux.fit_image(maxsma=cutout_halfwidth, step=step, linear=linear,
                                                  maxrit=cutout_halfwidth / 3)
                    if len(fitting_list) > 0:
                        return fitting_list

    except KeyboardInterrupt:
        sys.exit(1)
    except OverflowError:
        fail_count += 1
        if fail_count >= max_fails:
            return []

    return fitting_list


def extract_profiles(cutout_list, config, progress_bar=False, maxrit=None, isophote_testing=False):
    """
    Extract all available profiles
    :param cutout_list: A 2D list of cutouts. The length of each column needs to be the same!
    :param config: Configuration parameters
    :param progress_bar: Include a fancy progress bar with tqdm if set to True
    :param isophote_testing: Test whether isophote is crashing by passing fewer exception clauses.
        Should ONLY be used by developers working on the isophote code.
    :return:
    """

    output_profiles = []
    for i in cutout_list:
        output_profiles.append([])

    def run_model(index):
        # Iterate through each available object
        local_profiles = []
        for j in range(0, len(cutout_list)):
            try:
                if isophote_testing:
                    t = _isophote_fitting_tester(cutout_list[j][index], config)
                else:
                    t = isophote_fitting(cutout_list[j][index], config)
            except TimeoutException:
                continue
            if len(t) > 0:
                local_profiles.append(t.to_table())

        # If we extracted a profile of the model in each instance, save it
        if len(local_profiles) == len(cutout_list):
            for k in range(0, len(cutout_list)):
                output_profiles[k].append(local_profiles[k])

    # Iterate through each available object
    if progress_bar:
        for i in tqdm(range(0, len(cutout_list[0])), desc="Object"):
            run_model(i)
    else:
        for i in range(0, len(cutout_list[0])):
            run_model(i)

    return output_profiles


def extract_profiles_single_row(cutouts, config, bg_info=None):
    """
    Extract profiles for a single row.
    :param cutouts: A list of cutouts to extract. (Single row)
    :param config: Configuration parameters
    :param bg_info: Background info for the bg-added cutout (to maintain proper order in multithreading).
    :return:
    """

    output_profiles = []

    for i in range(0, len(cutouts)):
        t = isophote_fitting(cutouts[i], config)

        if len(t) > 0:
            output_profiles.append(t.to_table())

    if len(output_profiles) == len(cutouts):
        return output_profiles, bg_info
    else:
        return [], None


class TimeoutException(Exception):
    pass


def TimeoutHandler(signum, frame):
    raise TimeoutException
