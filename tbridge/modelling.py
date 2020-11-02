from astropy.io import fits
from astropy.modeling.models import Sersic2D

from numpy import exp, isnan, mgrid, ceil, pi
from numpy.random import choice, randint, uniform

from photutils.datasets import make_noise_image

from scipy.signal import convolve2d
from scipy.special import gamma
from scipy.stats import gaussian_kde

import tbridge


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

    sersic_models = []
    for i in range(n_models):
        fails, clean = 0, False
        this_r50, this_n, this_ellip, this_mag, this_r50_pix, i_r50 = None, None, None, None, None, None
        while not clean:
            if fails > 10000:
                break
            obj_choice = randint(0, len(mags))
            this_mag = mags[obj_choice]
            this_r50 = r50s[obj_choice]  # This is in arcseconds
            this_r50_pix = this_r50 / arc_conv  # This is in pixels
            this_n = ns[obj_choice]
            this_ellip = ellips[obj_choice]

            # Get the intensity at the half-light radius.
            i_r50 = i_at_r50(mag=this_mag, n=this_n, r_50=this_r50_pix)
            clean = not (i_r50 > 100) and not (isnan(i_r50)) and (this_n > 0.65) and (0 < this_ellip < 1) \
                    and (this_r50_pix > 0)
            if not clean:
                fails += 1
        if fails > 10000:
            raise ObjectGenError()

        sersic_model = Sersic2D(amplitude=i_r50, r_eff=this_r50_pix, n=this_n,
                                ellip=this_ellip, theta=uniform(0, 2 * pi),
                                x_0=cutout_size / 2, y_0=cutout_size / 2)
        z = sersic_model(x, y)
        sersic_models.append(z)

    return sersic_models


def add_to_locations_simple(models, config_values):

    image_dir, psf_filename = config_values["IMAGE_DIRECTORY"], config_values["PSF_FILENAME"]

    with fits.open(psf_filename) as psfs:
        image_filenames = tbridge.get_image_filenames(image_dir)

        convolved_models, bg_added_models = [], []

        for i in range(0, len(models)):
            model_width = models[i].shape[0]
            model_halfwidth = ceil(model_width / 2)

            image_filename = choice(image_filenames)

            image = tbridge.select_image(image_filename)
            image_wcs = tbridge.get_wcs(image_filename)

            if image is None:
                continue

            c_x = randint(model_width, image.shape[0] - model_width)
            c_y = randint(model_width, image.shape[1] - model_width)
            x_min, x_max = int(c_x - model_halfwidth), int(c_x + model_halfwidth)
            y_min, y_max = int(c_y - model_halfwidth), int(c_y + model_halfwidth)

            image_cutout = image[x_min: x_max - 1, y_min: y_max - 1]

            ra, dec = image_wcs.wcs_pix2world(c_x, c_y, 0)
            psf = tbridge.get_closest_psf(psfs, ra, dec).data

            # print(model_halfwidth, image.shape, type(image_wcs), c_x, c_y, psf.shape)

            convolved_model = convolve2d(models[i], psf, mode='same')
            bg_added_model = convolved_model + image_cutout

            convolved_models.append(convolved_model)
            bg_added_models.append(bg_added_model)

    return convolved_models, bg_added_models


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
        for model in models:
            index = randint(0, len(psfs))
            convolved_models.append(convolve2d(model, psfs[index].data, mode='same'))
        psfs.close()
    elif psf is not None:
        convolved_models = []
        for model in models:
            convolved_models.append(convolve2d(model, psf, mode='same'))
    else:
        return None

    return convolved_models


def add_to_provided_backgrounds(models, backgrounds):
    bgadded_models = []

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
    for model in models:
        noise_image = make_noise_image((model.shape[0], model.shape[1]), mean=bg_mean, stddev=bg_std)
        noisy_images.append(model + noise_image)
    return noisy_images


class ObjectGenError(Exception):
    """ Error raised if fails exceed 10000"""

    def __init__(self, expression="", message="Fails exceeded the limit"):
        self.expression = expression
        self.message = message
