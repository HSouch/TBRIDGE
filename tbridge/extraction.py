from numpy import max, pi
from numpy import ndarray, log
from numpy import unravel_index, argmax, ceil
from photutils import data_properties
from photutils.isophote import Ellipse, EllipseGeometry


def isophote_fitting(data: ndarray, linear=True):
    """
    Generates a table of results from isophote fitting analysis. This uses photutils Isophote procedure, which is
    effectively IRAF's Ellipse() method.
    Iterates over many possible input ellipses to force a higher success rate.
    :return:  The table of results, or an empty list if not fitted successfully.
    """
    # Set-up failsafe in case of strange infinte loops in photutils
    fail_count, max_fails = 0, 10

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
                fitting_list = flux.fit_image(maxsma=cutout_halfwidth, step=1., linear=True)
                if len(fitting_list) > 0:
                    return fitting_list
                geometry = EllipseGeometry(float(centre[0]), float(centre[1]), eps=0.5, sma=sma, pa=angle * pi / 180.)
                flux = Ellipse(data, geometry)
                fitting_list = flux.fit_image(maxsma=cutout_halfwidth, step=1., linear=True)
                if len(fitting_list) > 0:
                    return fitting_list

                geometry = EllipseGeometry(float(centre[0]), float(centre[1]), eps=0.3, sma=sma, pa=angle * pi / 180.)
                flux = Ellipse(data, geometry)
                fitting_list = flux.fit_image(maxsma=cutout_halfwidth, step=1., linear=True)
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
