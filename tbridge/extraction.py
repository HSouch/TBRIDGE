from numpy import max, pi
from numpy import ndarray, log
from numpy import unravel_index, argmax, ceil
from photutils import data_properties
from photutils.isophote import Ellipse, EllipseGeometry
try:
    from .isophote_l import Ellipse, EllipseGeometry
except ImportError:
    pass

from tqdm import tqdm

import warnings


def isophote_fitting(data: ndarray, linear=True):
    """
    Generates a table of results from isophote fitting analysis. This uses photutils Isophote procedure, which is
    effectively IRAF's Ellipse() method.
    Iterates over many possible input ellipses to force a higher success rate.
    :return:  The table of results, or an empty list if not fitted successfully.
    """
    # Set-up failsafe in case of strange infinte loops in photutils
    # warnings.filterwarnings("error")

    fail_count, max_fails = 0, 10000

    # Get centre of image and cutout halfwidth
    centre = unravel_index(argmax(data), data.shape)
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
        fitting_list = flux.fit_image(maxit=100, maxsma=cutout_halfwidth, step=1., linear=linear)
        if len(fitting_list) > 0:
            return fitting_list
    except RuntimeError:
        fail_count += 1
        if fail_count >= max_fails:
            return []
    except:
        fail_count += 1
        if fail_count >= max_fails:
            return []

    # If that fails, test a parameter space of starting ellipses
    try:
        for angle in range(0, 180, 45):
            for sma in range(1, 26, 5):
                geometry = EllipseGeometry(float(centre[0]), float(centre[1]), eps=0.9, sma=sma, pa=angle * pi / 180.)
                flux = Ellipse(data, geometry)
                fitting_list = flux.fit_image(maxsma=cutout_halfwidth, step=1., linear=linear)
                if len(fitting_list) > 0:
                    return fitting_list
                geometry = EllipseGeometry(float(centre[0]), float(centre[1]), eps=0.5, sma=sma, pa=angle * pi / 180.)
                flux = Ellipse(data, geometry)
                fitting_list = flux.fit_image(maxsma=cutout_halfwidth, step=1., linear=linear)
                if len(fitting_list) > 0:
                    return fitting_list

                geometry = EllipseGeometry(float(centre[0]), float(centre[1]), eps=0.3, sma=sma, pa=angle * pi / 180.)
                flux = Ellipse(data, geometry)
                fitting_list = flux.fit_image(maxsma=cutout_halfwidth, step=1., linear=linear)
                if len(fitting_list) > 0:
                    return fitting_list
    except RuntimeError:
        fail_count += 1
        if fail_count >= max_fails:
            return []
    except IndexError:
        fail_count += 1
        if fail_count >= max_fails:
            return []
    except:
        fail_count += 1
        if fail_count >= max_fails:
            return []

    return fitting_list


def extract_profiles(cutout_list, progress_bar=False, linear=True):
    """
    Extract all available profiles
    :param cutout_list: A 2D list of cutouts. The length of each column needs to be the same!
    :param progress_bar: Include a fancy progress bar with tqdm if set to True
    :param linear: Run the isophote fitting in linear mode
    :return:
    """

    output_profiles = []
    for i in cutout_list:
        output_profiles.append([])

    def run_model(index):
        # Iterate through each available object
        local_profiles = []
        for j in range(0, len(cutout_list)):

            t = isophote_fitting(cutout_list[j][index], linear=linear)
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
