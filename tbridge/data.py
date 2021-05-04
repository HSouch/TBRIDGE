"""

This module contains all general-purpose functions for loading and saving data. This includes pulling data out of
FITS files, saving TBriDGE simulation outputs, and more.
"""

import tbridge
import os
from pathlib import Path

from astropy.io import fits
from astropy.wcs import wcs
from astropy.table import Table

from numpy import arange, array, sqrt, str, save, load, ceil, zeros, isnan
from numpy.random import choice, uniform, randint


def get_closest_psf(psfs, obj_ra, obj_dec):
    """ Get the closest psf for a given object's RA and DEC"""

    def dist(ra1, ra2, dec1, dec2):
        return sqrt((ra1 - ra2) ** 2 + (dec1 - dec2) ** 2)

    shortest_dist = 999999
    closest_psf = None

    for psf in psfs:
        head = psf.header
        psf_ra, psf_dec = head["RA"], head["DEC"]
        local_dist = dist(obj_ra, psf_ra, obj_dec, psf_dec)
        if local_dist < shortest_dist:
            shortest_dist = local_dist
            closest_psf = psf

    return closest_psf


def get_wcs(fits_filename):
    """ Finds and returns the WCS for an image. If Primary Header WCS no good, searches each index until a good one
        is found. If none found, raises a ValueError
    """
    # Try just opening the initial header
    wcs_init = wcs.WCS(fits_filename)
    ra, dec = wcs_init.axis_type_names
    if ra.upper() == "RA" and dec.upper() == "DEC":
        return wcs_init

    else:
        hdu_list = fits.open(fits_filename)
        for n in hdu_list:
            try:
                wcs_slice = wcs.WCS(n.header)
                ra, dec = wcs_slice.axis_type_names
                if ra.upper() == "RA" and dec.upper() == "DEC":
                    return wcs_slice
            except:
                continue
        hdu_list.close()

    raise ValueError


def get_image_filenames(images_directory, image_band="i", check_band=False):
    """
    Retrieves a list of all available filenames for a given directory, and a given band.
    WARNING: Optimized for HSC filename format (ex: HSC-I_9813_4c3.fits).
    """
    image_filenames = []
    images = Path(images_directory).rglob('*.fits')
    for image in images:
        image = str(image)
        if check_band:
            image_no_path = image.split("/")[len(image.split("/")) - 1]
            filename_band = image_no_path.split("_")[0].split("-")[1].lower()
            if filename_band == image_band:
                image_filenames.append(image)
        else:
            image_filenames.append(image)
    return image_filenames


def select_image(filename):
    """ Opens an image-based FITS file and returns the first available image.

    Args:
        filename (str): The FITS filename and path.

    Returns:
        numpy.ndarray: The image obtained from the FITS file. If none found, will return None.

    """
    with fits.open(filename) as HDUList:
        image = None
        for i in range(0, len(HDUList)):
            try:
                image = HDUList[i].data
                if image.shape[0] > 0 and image.shape[1] > 0:
                    break
            except:
                continue
    return image


def generate_file_prefix(bin_params):
    """ Use the bin params to generate a file prefix."""
    prefix = "bin_"
    for j in range(0, len(bin_params)):
        if (j + 1) % 2 != 0:
            prefix += str(bin_params[j]) + "-"
        else:
            prefix += str(bin_params[j]) + "_"
    return prefix


def save_profiles(profile_list, bin_info, out_dir, keys, bg_info=None, cutout_info=None, structural_params=None):
    """ Saves a set of profiles into a properly formatted output directory, with proper filename format.

    Args:
        profile_list: The list of profiles (shape is m x n, where m is the number of different models for each
            object and n is the number of objects i.e. m rows of profiles from n objects.
        bin_info (arr): bin information for profile formatting (Obtained using the bin_params attribute from
            the tbridge.binning.Bin object).
        out_dir (str): the output directory to save the files to.
        keys (arr): the keys to generate subdirectory and file names with.
        bg_info (numpy.ndarray): Array of associated background info for the extracted profiles. Saves to its own
                    subdirectory.
        structural_params (astropy.table.Table): Astropy table of associated structural parameters
    """

    # Generate output structure
    subdirs = []
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for key in keys:
        subdir = key + "_profiles/"
        if not os.path.isdir(out_dir + subdir):
            os.mkdir(out_dir + subdir)
        subdirs.append(subdir)

    filename_prefix = generate_file_prefix(bin_info)
    valid_colnames = ["sma", "intens", "intens_err", "ellipticity", "ellipticity_err", "pa", "pa_err"]

    # Save profiles as FITS HDULists in the proper directories
    for i in range(0, len(profile_list)):
        profiles = profile_list[i]
        out_filename = filename_prefix + keys[i] + ".fits"
        out_hdulist = fits.HDUList()
        for prof in profiles:
            out_hdulist.append(fits.BinTableHDU(Table([prof[col] for col in valid_colnames],
                                                      names=valid_colnames)))

        out_hdulist.writeto(out_dir + subdirs[i] + out_filename, overwrite=True)

    # Save the additional available information
    if bg_info is not None:
        if not os.path.isdir(out_dir + "bg_info/"):
            os.mkdir(out_dir + "bg_info/")
        tbridge.save_array(bg_info, out_dir + "bg_info/" + filename_prefix + "bgs.npy")
    if cutout_info is not None:
        if not os.path.isdir(out_dir + "cutout_info/"):
            os.mkdir(out_dir + "cutout_info/")
        tbridge.save_array(cutout_info, out_dir + "cutout_info/" + filename_prefix + "bgs.npy")

    if structural_params is not None:
        if not os.path.isdir(out_dir + "params/"):
            os.mkdir(out_dir + "params/")
        structural_params.write(out_dir + "params/" + tbridge.generate_file_prefix(bin_info) + "params.fits",
                                format="fits", overwrite=True)


def save_profile_set(profiles, out_filename="profiles.fits"):
    """ Save an array of profiles to a FITS file.

    Args:
        profiles: List of profile tables.
        out_filename: Filename to save FITS file to.

    """

    valid_colnames = ["sma", "intens", "intens_err", "ellipticity", "ellipticity_err", "pa", "pa_err"]
    out_hdulist = fits.HDUList()
    for prof in profiles:
        out_hdulist.append(fits.BinTableHDU(Table([prof[col] for col in valid_colnames],
                               names=valid_colnames)))

    out_hdulist.writeto(out_filename, overwrite=True)


def load_profile_set(filename):
    """ Load a set of profiles from a FITS file specified by the filename"""
    tables = []
    with fits.open(filename) as HDUList:
        for hdu in HDUList:
            try:
                tables.append(Table.read(hdu))
            except:
                continue
    return tables


def save_cutouts(cutouts, output_filename="cutouts.fits"):
    """ Save a set of cutouts to a fits HDUList object """
    out_hdulist = fits.HDUList()

    for cutout in cutouts:
        out_hdulist.append(fits.ImageHDU(data=cutout))

    out_hdulist.writeto(output_filename, overwrite=True)


def load_cutouts(filename):
    """ Load in a list of cutouts from the given filename (will try all hdus in the HDUList)"""
    cutouts = []
    HDUList = fits.open(filename)
    for n in HDUList:
        try:
            image = n.data
            if image.shape[0] > 0 and image.shape[1] > 0:
                cutouts.append(image)
        except:
            continue

    HDUList.close()

    return cutouts


def tables_from_file(filename):
    """
    Load a set of tables from a given HDUList.
    :param filename: filename to gather tables from.
    :return:
    """
    tables = []

    HDUList = fits.open(filename)
    for n in HDUList:
        try:
            t = Table.read(n)
            tables.append(t)
        except:
            continue
    HDUList.close()

    return tables


def params_from_filename(filename):
    """ Return the parameters inferred from the input filename (based on TBRIDGE formatting)."""
    filename_no_path = filename.split("/")[len(filename.split("/")) - 1]
    splits = filename_no_path.split("_")
    splits = splits[1:len(splits) - 1]
    params = []
    for split in splits:
        low_high = split.split("-")
        params.append((float(low_high[0]) + float(low_high[1])) / 2)
    return params


def save_array(arr, filename="arr.npy"):
    """ Save an array (npy binary format)"""
    save(filename, arr)


def load_array(filename):
    """ Load an array """
    try:
        return load(filename)
    except FileNotFoundError:
        print("File: ", filename, " not found. Returning NoneType")
        return None


def get_backgrounds(config_values, n=50, return_psfs=True, return_bg_info=True):
    """ Retrieve a random set of backgrounds from the input image directory.
    Args:
        config_values: User provided configuration file.
        n: Number of backgrounds to generate.
        return_psfs (bool): Return the psfs associated with the backgrounds
        return_bg_info (bool): Return the background info.
    :return:
    """

    image_dir, psf_filename = config_values["IMAGE_DIRECTORY"], config_values["PSF_FILENAME"]
    size = config_values["SIZE"]

    with fits.open(psf_filename) as psfs:
        image_filenames = tbridge.get_image_filenames(image_dir)

        psf_list, bg_added_models, image_list, ras, decs = [], [], [], [], []

        bg_infotable = {"IMAGES": [], "RAS": [], "DECS": [], "XS": [], "YS": []}

        for i in range(0, n):
            model_halfwidth = ceil(size / 2)

            image_filename = choice(image_filenames)

            image = tbridge.select_image(image_filename)
            image_wcs = tbridge.get_wcs(image_filename)

            if image is None:
                continue

            c_x = randint(model_halfwidth + 1, image.shape[0] - model_halfwidth - 1)
            c_y = randint(model_halfwidth + 1, image.shape[1] - model_halfwidth - 1)
            x_min, x_max = int(c_x - model_halfwidth), int(c_x + model_halfwidth)
            y_min, y_max = int(c_y - model_halfwidth), int(c_y + model_halfwidth)

            image_cutout = image[x_min: x_max - 1, y_min: y_max - 1]
            ra, dec = image_wcs.wcs_pix2world(c_x, c_y, 0)

            if return_psfs:
                psf = tbridge.get_closest_psf(psfs, ra, dec).data
                psf_list.append(psf)

            bg_added_models.append(image_cutout)
            bg_infotable["IMAGES"].append(image_filename)
            bg_infotable["RAS"].append(ra)
            bg_infotable["DECS"].append(dec)
            bg_infotable["XS"].append(c_x)
            bg_infotable["YS"].append(c_y)

    if return_psfs:
        if return_bg_info:
            return bg_added_models, psf_list, bg_infotable
        else:
            return bg_added_models, psf_list
    elif return_bg_info:
        return bg_added_models, bg_infotable
    else:
        return bg_added_models


def as_dir(directory):
    """ Add a forward slash if one is not at the end of a string. """
    if directory[-1] != '/':
        return directory + "/"
    else:
        return directory


def cutout_stitch(cutouts, masked_cutouts=None, output_filename=None):
    """ Generate a full stitch from a set of cutouts.

    Args:
        cutouts: Array of cutouts (ideally unmasked but it makes no difference). Need to be all the same size as the
            first cutout.
        masked_cutouts: The same cutouts but masked
        output_filename: Optional output to save as HDUList

    Returns:
        Returns either the stitch, or a tuple of the stich and the mask stitch if masked_cutouts is submitted..
    """

    i = int(sqrt(len(cutouts)))
    j = int(len(cutouts) / i)

    c_width, c_height = cutouts[0].shape[0], cutouts[1].shape[1]
    w, h = i * c_width, j * c_height

    canvas = zeros((w, h))

    if masked_cutouts is not None:
        masked_canvas = zeros((w, h))
    else:
        masked_canvas = None

    n = 0

    # Add cutouts to the canvas
    for x in range(i):
        for y in range(j):
            try:
                canvas[x * c_width: x * c_width + c_width,
                y * c_height: y * c_height + c_height] = cutouts[n]
            except ValueError:
                continue
            if masked_cutouts is not None:
                masked_cutout = masked_cutouts[n]
                mask = isnan(masked_cutout).astype(int)

                masked_canvas[x * c_width: x * c_width + c_width,
                y * c_height: y * c_height + c_height] = mask
            n += 1
            if n == len(cutouts):
                break

    if output_filename is not None:
        HDUList = fits.HDUList()
        HDUList.append(fits.ImageHDU(canvas))
        if masked_cutouts is not None:
            HDUList.append(fits.ImageHDU(masked_canvas))
        HDUList.writeto(output_filename, overwrite=True)

    if masked_cutouts is None:
        return canvas
    else:
        return canvas, masked_canvas
