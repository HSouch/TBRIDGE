import os
from pathlib import Path

from astropy.io import fits
from astropy.wcs import wcs
from astropy.table import Table

from numpy import arange, array, sqrt, str
from numpy.random import choice, uniform


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


def generate_output_structure(out_dir):
    """
    Generates the structure of the output filesystem, with outdir being the top-level directory.
    :param out_dir:
    :return:
    """

    bare_profile_outdir = out_dir + "bare_profiles/"
    bgadded_profile_outdir = out_dir + "bgadded_profiles/"
    noisy_outdir = out_dir + "noisy_profiles/"
    psf_outdir = out_dir + "psf_profiles/"
    localsub_outdir = out_dir + "localsub_profiles/"

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for directory in (bare_profile_outdir, bgadded_profile_outdir, noisy_outdir, psf_outdir, localsub_outdir):
        if not os.path.isdir(directory):
            os.mkdir(directory)

    return bare_profile_outdir, bgadded_profile_outdir, noisy_outdir, localsub_outdir, psf_outdir


def generate_output_report(out_dir="", t_final=0., t_init=0., catalog_filename=""):
    """
    Generates an output report to the user's specifications.
    :param out_dir:
    :param t_final:
    :param t_init:
    :param catalog_filename:
    :return:
    """
    output_report = open(out_dir + "output_log.txt", "w")
    lines = []
    lines.append("Time (seconds): " + str(t_final - t_init) + "\n")
    lines.append("Time (minutes): " + str((t_final - t_init) / 60) + "\n")
    lines.append("\n")
    lines.append("Catalog: " + str(catalog_filename) + "\n")
    output_report.writelines(lines)


def save_profiles(profile_list, bin_info, outdir, keys):
    """
    Saves a set of profiles into a properly formatted output directory, with proper filename format.
    :param profile_list: The list of profiles (shape is m x n, where m is the number of different models for each
    object and n is the number of objects i.e. m rows of profiles from n objects.
    :param bin_info: bin information for profile formatting.
    :param outdir: the output directory to save the files to.
    :param keys: the keys to generate subdirectory and file names with.
    :return:
    """
    def generate_file_prefix(bin_params):
        prefix = "bin_"
        for j in range(0, len(bin_params)):
            if (j + 1) % 2 != 0:
                prefix += str(bin_params[j]) + "-"
            else:
                prefix += str(bin_params[j]) + "_"
        return prefix

    # Generate output structure
    subdirs = []
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for key in keys:
        subdir = key + "_profiles/"
        if not os.path.isdir(outdir + subdir):
            os.mkdir(outdir + subdir)
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

        out_hdulist.writeto(outdir + subdirs[i] + out_filename, overwrite=True)

    return None


def save_profile_set(profiles, out_filename="profiles.fits"):
    """
    Save a set of profiles to a FITS file.
    :param profiles: List of profile tables.
    :param out_filename: Filename to save FITS file to.
    :return:
    """
    valid_colnames = ["sma", "intens", "intens_err", "ellipticity", "ellipticity_err", "pa", "pa_err"]
    out_hdulist = fits.HDUList()
    for prof in profiles:
        out_hdulist.append(fits.BinTableHDU(Table([prof[col] for col in valid_colnames],
                               names=valid_colnames)))

    out_hdulist.writeto(out_filename, overwrite=True)


def load_profile_set(filename):
    """ Load a set of profiles from a FITS file """
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


def cutouts_from_file(filename):
    """ Load in a list of cutouts from a given filename (will try all hdus in the HDUList)"""
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


def trim_hdulist(input_filename, indices, output_filename="out.fits"):
    """
    Trims an HDUList based on a set of user-provided indices
    :param input_filename:
    :param indices:
    :param output_filename:
    :return: HDUList of size <= len(indices)

    USAGE
    indices = [1, 4, 5, 8, 9, 10]
    trim_hdulist("input.fits", indices, output_filename="output.fits")
    """

    HDUList = fits.open(input_filename)
    out_hdulist = fits.HDUList()
    print(len(HDUList))
    for n in range(0, len(HDUList)):
        if n in indices:
            out_hdulist.append(HDUList[n])

    out_hdulist.writeto(output_filename, overwrite=True)


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


def bin_index(val, bins):
    """ Get bin index for a given value and a set of bin parameters """
    for index in range(0, len(bins)):
        if val < bins[index]:
            return index
    return len(bins)
