import os
from pathlib import Path

from astropy.io import fits
from astropy.wcs import wcs
from numpy import arange, array, sqrt, str
from numpy.random import choice, randint, uniform


def get_closest_psf(psfs: fits.HDUList, obj_ra: float, obj_dec: float):
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
    WARNING: Optimized for HSC filenames (ex: HSC-I_9813_4c3.fits).
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


def get_tract_and_patch(filename):
    """ Returns the tract and patch (as strings) for a given image filename."""
    clip_1 = filename.split(".")[0]
    clip_2 = clip_1.split("_")
    tract, patch = clip_2[1], clip_2[2].replace("c", ",")
    return tract, patch


def random_selection(coverage_table, ra_min, ra_max, dec_min, dec_max, band="i"):
    """ Selects an image based on a random RA and DEC selection. """
    for n in range(1000):
        ra, dec = uniform(ra_min, ra_max), uniform(dec_min, dec_max)

        band_rows = []
        for row in coverage_table:
            filename = row["Image Filename"]
            filename_band = filename.split("/")[len(filename.split("/")) - 1].split("_")[0].split("-")[1].lower()
            if filename_band == band:
                band_rows.append(row)

        for row in band_rows:
            if row["ra_2"] < ra < row["ra_1"] and row["dec_1"] < dec < row["dec_2"]:
                return row["Image Filename"]


def load_positions(location_table, n=100, ra_key="RA", dec_key="DEC", img_filename_key="img_filename",
                   check_band=False, band_key="band", band="i"):
    images, ras, decs = location_table[img_filename_key], location_table[ra_key], location_table[dec_key]

    if check_band:
        bands, band_mask = location_table[band_key], []
        for i in range(0, len(bands)):
            if str(bands[i]) == band:
                band_mask.append(True)
            else:
                band_mask.append(False)
        band_mask = array(band_mask)

        images, ras, decs = images[band_mask], ras[band_mask], decs[band_mask]

    index_array = arange(0, len(images), 1, dtype=int)
    indices = choice(index_array, n, replace=True)

    images, ras, decs = images[indices], ras[indices], decs[indices]

    return array(images, dtype=str), array(ras), array(decs)


def select_image(filename):
    image, HDUList = None, fits.open(filename)

    for i in range(0, 100):
        try:
            index = randint(0, len(HDUList))
            image = HDUList[index].data
            if image.shape[0] > 0 and image.shape[1] > 0:
                break
        except:
            print("Error in retrieving this image")
            continue
    HDUList.close()

    return image


def generate_output_structure(outdir):
    """
    Generates the structure of the output filesystem, with outdir being the top-level directory.
    :param outdir:
    :return:
    """

    bare_profile_outdir = outdir + "bare_profiles/"
    bgadded_profile_outdir = outdir + "bgadded_profiles/"
    noisy_outdir = outdir + "noisy_profiles/"
    psf_outdir = outdir + "psf_profiles/"
    localsub_outdir = outdir + "localsub_profiles/"

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for directory in (bare_profile_outdir, bgadded_profile_outdir, noisy_outdir, psf_outdir, localsub_outdir):
        if not os.path.isdir(directory):
            os.mkdir(directory)

    return bare_profile_outdir, bgadded_profile_outdir, noisy_outdir, localsub_outdir, psf_outdir


def generate_output_report(outdir="", t_final=0., t_init=0., catalog_filename=""):
    """
    Generates an output report to the user's specifications.
    :param outdir:
    :param t_final:
    :param t_init:
    :param catalog_filename:
    :return:
    """
    output_report = open(outdir + "output_log.txt", "w")
    lines = []
    lines.append("Time (seconds): " + str(t_final - t_init) + "\n")
    lines.append("Time (minutes): " + str((t_final - t_init) / 60) + "\n")
    lines.append("\n")
    lines.append("Catalog: " + str(catalog_filename) + "\n")
    output_report.writelines(lines)
