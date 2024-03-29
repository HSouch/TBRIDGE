from astropy.io import fits
from astropy.modeling.models import Gaussian2D, Sersic2D
from astropy.modeling import Fittable2DModel, Parameter
from astropy.stats import sigma_clipped_stats

from numpy import exp, isnan, mgrid, ceil, pi, cosh, cos, sin, sqrt, std, ndarray
from numpy.random import choice, randint, uniform

from photutils.datasets import make_noise_image

from scipy.signal import convolve2d
from scipy.special import gamma, gammainc, gammaincinv, kn
from scipy.optimize import newton
from scipy.stats import gaussian_kde

import tbridge


def b(n, estimate=False):
    """ Get the b_n normalization constant for the sersic profile. From Graham and Driver."""
    if estimate:
        return 2 * n - (1 / 3) + (4 / (405 * n)) + (46 / (25515 * (n ** 2)))
    else:
        return gammaincinv(2 * n, 0.5)


def core_sersic_b(n, r_b, r_e):
    """
    Get the scale parameter b using the Newton-Raphson root finder.
    :param n: Sersic index
    :param r_b: Break radius
    :param r_e: Effective radius
    :return:
    """
    # Start by getting an initial guess at b (using the regular Sersic estimation)
    b_guess = 2 * n - (1 / 3)

    # Define the combination of gamma functions that makes up the relation
    # We want to find the zeroes of this.
    def evaluate(b_in):
        comp1 = gamma(2 * n)
        comp2 = gammainc(2 * n, b_in * ((r_b / r_e) ** (1 / n)))
        comp3 = 2 * gammainc(2 * n, b_in)

        return comp1 + comp2 - comp3

    return newton(evaluate, x0=b_guess)


def i_at_r50(mag, n=2, r_50=2, m_0=27):
    """ Get the intensity at the half-light radius """
    b_n = b(n)
    l_tot = 10 ** ((mag - m_0) / -2.5) * (b_n ** (2 * n))
    denom = (r_50 ** 2) * 2 * pi * n * exp(b_n) * gamma(2 * n)
    i_e = l_tot / denom

    return i_e


def pdf_resample(columns, resample_size=1000):
    """
    Makes a multidimensional probability distribution function (PDF) for galaxy structural parameters.
    :param columns: columns to resample
    :param resample_size: number of objects to resample
    :return:
    """

    obj_resample = gaussian_kde(columns).resample(resample_size)

    return obj_resample


def structural_parameters(keys, columns):
    """
    Gets the structural parameters in a specific order (if catalog is generated by the method in tbridge.binning)
    """
    mags = columns[keys.index("MAGS")]
    r50s = columns[keys.index("R50S")]
    ns = columns[keys.index("NS")]
    ellips = columns[keys.index("ELLIPS")]

    return mags, r50s, ns, ellips


def simulate_sersic_models(mags, r50s, ns, ellips, config_values, n_models=10):
    """
    Simulates Sersic models WITHOUT CONVOLUTION
    :param mags: object magnitudes
    :param r50s: object half-light radii (in arcseconds)
    :param ns: object Sersic indices
    :param ellips: object ellipticities
    :param config_values: configuration file information
    :param n_models: number of models to generate
    :return:
    """

    cutout_size, arc_conv = config_values["SIZE"], config_values["ARC_CONV"]

    # Prep mgrid
    x, y = mgrid[:cutout_size, :cutout_size]

    # Resample parameters and get a better distribution.
    mags, r50s, ns, ellips = tbridge.pdf_resample((mags, r50s, ns, ellips), 1000)

    sersic_models, param_dicts = [], []
    for i in range(n_models):
        fails, clean = 0, False
        this_r50, this_n, this_ellip, this_mag, this_r50_pix, i_r50 = None, None, None, None, None, None
        while not clean:
            if fails > 10000:
                break
            obj_choice = randint(0, len(mags),)
            this_mag = mags[obj_choice]
            this_r50 = r50s[obj_choice]  # This is in arcseconds
            this_r50_pix = this_r50 / arc_conv  # This is in pixels
            this_n = ns[obj_choice]
            this_ellip = ellips[obj_choice]

            # Get the intensity at the half-light radius.
            i_r50 = i_at_r50(mag=this_mag, n=this_n, r_50=this_r50_pix, m_0=config_values["ZEROPOINT"])
            clean = not (i_r50 > 100) and not (isnan(i_r50)) and (this_n > 0.65) and (0 < this_ellip < 1) \
                    and (this_r50_pix > 0)
            if not clean:
                fails += 1
            if fails > 10000:
                continue

        pa = uniform(0, 2 * pi)
        struct_param_dict = {"I_R50": i_r50,
                             "R50": this_r50_pix,
                             "N": this_n,
                             "ELLIP": this_ellip,
                             "PA": pa,
                             "MAG": this_mag}

        sersic_model = Sersic2D(amplitude=i_r50, r_eff=this_r50_pix, n=this_n,
                                ellip=this_ellip, theta=pa,
                                x_0=cutout_size / 2, y_0=cutout_size / 2)
        z = sersic_model(x, y)
        sersic_models.append(z)
        param_dicts.append(struct_param_dict)

    return sersic_models, param_dicts


def add_to_background(model, config, convolve=True, return_bg_info=False, threshold=1e-4):
    """
    Add a model to a given background
    """
    image_dir, psf_filename = config["IMAGE_DIRECTORY"], config["PSF_FILENAME"]
    image_filenames = tbridge.get_image_filenames(image_dir)
    bg_infotable = {"IMAGES": [], "RAS": [], "DECS": [], "XS": [], "YS": []}

    with fits.open(psf_filename) as psfs:
        model_width = model.shape[0]
        model_halfwidth = ceil(model_width / 2)

        image_filename = choice(image_filenames)

        fail_counter = 0

        while 1:
            fail_counter += 1
            if fail_counter > 50:
                return (None, None) if convolve else None

            image = tbridge.select_image(image_filename)
            image_wcs = tbridge.get_wcs(image_filename)

            c_x = randint(model_halfwidth + 1, image.shape[0] - model_halfwidth - 1)
            c_y = randint(model_halfwidth + 1, image.shape[1] - model_halfwidth - 1)
            x_min, x_max = int(c_x - model_halfwidth), int(c_x + model_halfwidth)
            y_min, y_max = int(c_y - model_halfwidth), int(c_y + model_halfwidth)

            image_cutout = image[x_min: x_max - 1, y_min: y_max - 1]

            # Check if the background std is less than a given threshold
            if threshold is not None:
                bg_mean, bg_median, bg_std = sigma_clipped_stats(image_cutout, sigma=3.)
                if bg_std < threshold:
                    continue
                else:
                    break

        ra, dec = image_wcs.wcs_pix2world(c_x, c_y, 0)
        psf = tbridge.get_closest_psf(psfs, ra, dec).data

        if return_bg_info:
            bg_infotable["IMAGES"].append(image_filename)
            bg_infotable["RAS"].append(ra)
            bg_infotable["DECS"].append(dec)
            bg_infotable["XS"].append(c_x)
            bg_infotable["YS"].append(c_y)


        if convolve:
            convolved = convolve2d(model, psf, mode='same')
            bg_added = convolved + image_cutout

            return bg_added, convolved
        else:
            bg_added = model + image_cutout
            return bg_added, None


def get_background(config, threshold=1e-4):
    """
        Get a valid background and background info
        """
    image_dir, psf_filename = config["IMAGE_DIRECTORY"], config["PSF_FILENAME"]
    image_filenames = tbridge.get_image_filenames(image_dir)
    bg_infotable = {}

    with fits.open(psf_filename) as psfs:
        model_width = config["SIZE"]
        model_halfwidth = ceil(model_width / 2)

        image_filename = choice(image_filenames)

        fail_counter = 0

        while 1:
            fail_counter += 1
            if fail_counter > 50:
                return None

            image = tbridge.select_image(image_filename)
            image_wcs = tbridge.get_wcs(image_filename)

            c_x = randint(model_halfwidth + 1, image.shape[0] - model_halfwidth - 1)
            c_y = randint(model_halfwidth + 1, image.shape[1] - model_halfwidth - 1)
            x_min, x_max = int(c_x - model_halfwidth), int(c_x + model_halfwidth)
            y_min, y_max = int(c_y - model_halfwidth), int(c_y + model_halfwidth)

            image_cutout = image[x_min: x_max - 1, y_min: y_max - 1]

            # Check if the background std is less than a given threshold
            if threshold is not None:
                bg_mean, bg_median, bg_std = tbridge.estimate_background_sigclip(image_cutout)
                if bg_std < threshold:
                    continue
                else:
                    break
        ra, dec = image_wcs.wcs_pix2world(c_x, c_y, 0)
        psf = tbridge.get_closest_psf(psfs, ra, dec).data

        bg_infotable["IMAGES"] = image_filename
        bg_infotable["RAS"] = ra
        bg_infotable["DECS"] = dec
        bg_infotable["XS"] = c_x
        bg_infotable["YS"] = c_y
        bg_infotable["BG_MEAN"] = bg_mean
        bg_infotable["BG_MEDIAN"] = bg_median
        bg_infotable["BG_STD"] = bg_std

    return image_cutout, psf, bg_infotable


def convolve_models(models, config_values=None, psf=None):
    """
    Convolve a set of models.
    :param models: List of cutouts to convolve.
    :param config_values:
    :param psf:
    :return:
    """

    if config_values is not None:
        psf_filename = config_values["PSF_FILENAME"]
        psfs = fits.open(psf_filename)
        convolved_models = []

        if type(models) == ndarray:
            index = randint(0, len(psfs))
            return convolve2d(models, psfs[index].data, mode='same')
        else:
            for model in models:
                index = randint(0, len(psfs))
                convolved_models.append(convolve2d(model, psfs[index].data, mode='same'))
        psfs.close()
    elif psf is not None:
        if type(models) == ndarray:
            return convolve2d(models, psf, mode='same')
        else:
            convolved_models = []
            for model in models:
                convolved_models.append(convolve2d(model, psf, mode='same'))
    else:
        return None

    return convolved_models


def add_to_provided_backgrounds(models, backgrounds):
    bgadded_models, bg_datatables = [], []

    if type(models) == ndarray:
        index = randint(0, len(backgrounds))

        return models + backgrounds[index]
    else:
        for model in models:
            index = randint(0, len(backgrounds))
            bgadded_models.append(model + backgrounds[index])

    return bgadded_models


def add_to_noise(models, bg_mean=0., bg_std=0.025):
    """
    Returns the models added to Gaussian noise with user-defined mean and standard deviation
    :param models: The models to add noise to.
    :param bg_mean: Mean of the Gaussian noise.
    :param bg_std: Standard deviation of the Gaussian noise.
    :return:
    """
    noisy_images = []

    if type(models) == ndarray:
        noise_image = make_noise_image((models.shape[0], models.shape[1]), mean=bg_mean, stddev=bg_std)
        return models + noise_image
    else:
        for model in models:
            noise_image = make_noise_image((model.shape[0], model.shape[1]), mean=bg_mean, stddev=bg_std)
            noisy_images.append(model + noise_image)
        return noisy_images


def simulate_bg_gaussians(n_bgs, n_models, width=251, noise_mean=0.001, noise_std=0.01,
                          amplitude=(0, 3), stddev=(2, 10), min_dist=0):
    x_0 = width /2
    x, y = mgrid[:width, :width]
    backgrounds = []
    for i in range(0, n_bgs):
        # Generate base image with Gaussian noise
        random_bg = make_noise_image((width, width), distribution='gaussian', mean=noise_mean, stddev=noise_std)

        # Generate a random assortment of Gaussian models with the supplied User parameters
        for j in range(0, n_models):
            x_mean, y_mean, dist = uniform(0, width), uniform(0, width), 0
            while dist <= min_dist:
                x_mean, y_mean = uniform(0, width), uniform(0, width)
                dist = sqrt((x_mean - x_0) ** 2 + (y_mean - x_0) ** 2)

            model = Gaussian2D(amplitude=uniform(amplitude[0], amplitude[1]),
                               x_mean=x_mean,
                               y_mean=y_mean,
                               x_stddev=uniform(stddev[0], stddev[1]),
                               y_stddev=uniform(stddev[0], stddev[1]),
                               theta=uniform(0, 2 * pi))
            z = model(x, y)
            random_bg += z
        backgrounds.append(random_bg)
    return backgrounds


class EdgeOnDisk(Fittable2DModel):
    """
    Two-dimensional Edge-On Disk model.

    Parameters
    ----------
    amplitude : float
        Brightness at galaxy centre
    scale_x : float
        Scale length along the semi-major axis
    scale_y : float
        Scale length along the semi-minor axis
    x_0 : float, optional
        x position of the center
    y_0 : float, optional
        y position of the center
    theta: float, optional
        Position angle in radians, counterclockwise from the
        positive x-axis.
    """
    amplitude = Parameter(default=1)
    scale_x = Parameter(default=1)
    scale_y = Parameter(default=0.5)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    theta = Parameter(default=0)

    @classmethod
    def evaluate(cls, x, y, amplitude, scale_x, scale_y, x_0, y_0, theta):
        """Exaluate model on a 2D x-y grid."""

        x_maj = abs((x - x_0) * cos(theta) + (y - y_0) * sin(theta))
        x_min = -(x - x_0) * sin(theta) + (y - y_0) * cos(theta)

        return amplitude * (x_maj / scale_x) * kn(1, x_maj / scale_x) / (cosh(x_min / scale_y) ** 2)


class Core_Sersic(Fittable2DModel):
    """
    Two-dimensional Edge-On Disk model.

    Parameters
    ----------
    r_e : float
        Effective radius of the galaxy.
    r_b : float
        Break Radius (Where the model switches from one regime to the other).
    I_b : float
        Intensity at the break radius
    alpha : float
        Defines the "sharpness" of the model transitions
    gamma : float
        Power law slope
    n     : float
        Sersic index (see info on Sersic profiles if needed)
    x_0 : float, optional
        x position of the center
    y_0 : float, optional
        y position of the center
    theta: float, optional
        Position angle in radians, counterclockwise from the positive x-axis.
    ellip: float, optional
        Ellipticity of the model (default is 0 : circular)


    """

    r_e = Parameter(default=5)
    r_b = Parameter(default=1)
    I_b = Parameter(default=1)
    n = Parameter(default=1)
    alpha = Parameter(default=1)
    gamma = Parameter(default=1)

    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    theta = Parameter(default=0)
    ellip = Parameter(default=0)

    @classmethod
    def evaluate(cls, x, y, r_e, r_b, I_b, n, alpha, gamma, x_0, y_0, theta, ellip):
        """Two dimensional Core-Sersic profile function."""

        bn = core_sersic_b(n, r_b, r_e)

        def core_sersic_i_prime():
            return I_b * (2 ** (- gamma / alpha)) * exp(bn * (2 ** (1 / (alpha * n))) * (r_b / r_e) ** (1 / n))

        i_prime = core_sersic_i_prime()

        cos_theta, sin_theta = cos(theta), sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        z = sqrt(x_maj ** 2 + (x_min / (1-ellip)) ** 2)

        comp_1 = i_prime * ((1 + ((r_b / z) ** alpha)) ** (gamma / alpha))
        comp_2 = -bn * (((z ** alpha) + (r_b ** alpha)) / (r_e ** alpha)) ** (1 / (n * alpha))

        return comp_1 * exp(comp_2)


class ObjectGenError(Exception):
    """ Error raised if fails exceed 10000"""

    def __init__(self, expression="", message="Fails exceeded the limit"):
        self.expression = expression
        self.message = message


