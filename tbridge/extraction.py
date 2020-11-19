from numpy import max, pi, count_nonzero, log
from numpy import unravel_index, argmax, ceil
from photutils import data_properties
from photutils.isophote import Ellipse, EllipseGeometry

try:
    from .isophote_l import Ellipse, EllipseGeometry
except ImportError:
    pass

from tqdm import tqdm

import sys
import signal


def isophote_fitting(data, config=None, use_alarm=False, alarm_time=60, centre_method='standard',
                     fit_method='standard', maxrit=None):
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

    if test_verbose:
        print("Verbose", verbose, test_verbose, "Linear:", linear, "Step:", step, "Use Alarm:", use_alarm,
              "Alarm Time:", alarm_time)

    # Get centre of image and cutout halfwidth
    if centre_method == 'standard':
        centre = (data.shape[0]/2, data.shape[1]/2)
    elif centre_method == 'max':
        centre = unravel_index(argmax(data), data.shape)
    else:
        centre = (data.shape[0] / 2, data.shape[1] / 2)

    cutout_halfwidth = max((ceil(data.shape[0] / 2), ceil(data.shape[1] / 2)))

    fitting_list = []

    if use_alarm:
        original_handler = signal.signal(signal.SIGALRM, TimeoutHandler)
        signal.alarm(alarm_time)
    else:
        original_handler = None

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
    except RuntimeError:
        fail_count += 1
        if fail_count >= max_fails:
            return []
    except TimeoutException:
        signal.signal(signal.SIGALRM, original_handler)
        if verbose:
            print("Timeout reached due to signal alarm")
        fail_count += 1
        if fail_count >= max_fails:
            return []
    except:
        fail_count += 1
        if fail_count >= max_fails:
            return []

    if use_alarm:
        signal.alarm(alarm_time)

    # If that fails, test a parameter space of starting ellipses
    try:
        for angle in range(0, 180, 45):
            for sma in range(1, 26, 5):
                for eps in (0.3, 0.5, 0.9):
                    geometry = EllipseGeometry(float(centre[0]), float(centre[1]), eps=eps,
                                               sma=sma, pa=angle * pi / 180.)
                    flux = Ellipse(data, geometry)
                    fitting_list = flux.fit_image(maxsma=cutout_halfwidth, step=step, linear=linear,
                                                  maxrit=cutout_halfwidth / 3)
                    if len(fitting_list) > 0:
                        if use_alarm:
                            signal.alarm(0)
                        return fitting_list

    except KeyboardInterrupt:
        sys.exit(1)
    except RuntimeError:
        fail_count += 1
        if fail_count >= max_fails:
            return []
    except IndexError:
        fail_count += 1
        if fail_count >= max_fails:
            return []
    except TimeoutException:
        signal.alarm(0)
        if verbose:
            print("Timeout reached due to signal alarm")
        fail_count += 1
        if fail_count >= max_fails:
            return []
    except:
        fail_count += 1
        if fail_count >= max_fails:
            return []

    if use_alarm:
        signal.alarm(0)

    return fitting_list


def extract_profiles(cutout_list, config, progress_bar=False, use_alarm=False, alarm_time=60, maxrit=None):
    """
    Extract all available profiles
    :param cutout_list: A 2D list of cutouts. The length of each column needs to be the same!
    :param config: Configuration parameters
    :param progress_bar: Include a fancy progress bar with tqdm if set to True
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
                t = isophote_fitting(cutout_list[j][index], config, use_alarm=use_alarm, alarm_time=alarm_time)
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


def extract_profiles_single_row(cutouts, config, bg_info=None,):
    """
    Extract profiles for a single row.
    :param cutouts: A list of cutouts to extract. (Single row)
    :param config: Configuration parameters
    :param bg_info: Background info for the bg-added cutout (to maintain proper order in multithreading).
    :return:
    """

    output_profiles = []

    for i in range(0, len(cutouts)):
        t = isophote_fitting(cutouts[i], config, use_alarm=False)

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
