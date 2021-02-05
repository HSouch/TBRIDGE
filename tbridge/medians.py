from numpy import arange, nan, inf, sort, median, floor, max, nanmedian
from numpy.random import choice
from scipy.interpolate import interp1d

import multiprocessing as mp

from astropy.io import fits
from astropy.table import Table

import tbridge
import os
import shutil
from pathlib import Path


def as_interpolations(profile_list, fill_value_type='min', x_key="sma", y_key="intens"):

    interps = []

    for prof in profile_list:
        sma, intens = prof[x_key], prof[y_key]
        try:
            interp = interp1d(sma, intens, bounds_error=False, fill_value=0)
            interps.append(interp)
        except ValueError:
            print("ValueError in getting interpolation")
            continue
    return interps


def bin_max(profile_list, key="sma"):
    max_val = -999

    for prof in profile_list:
        arr_max = max(prof[key])
        max_val = arr_max if arr_max > max_val else max_val

    return max_val


def get_median(pop, bin_max, ignore_nans=True, profile_division=100):
    """
    Obtain the median profile for a bin of profiles. Takes "slices" along the x-axis, obtaining all profile values at
    that slice, and populates a list of median values. Also obtains the upper and lower sigma values.

    :param pop: The list of profiles to get the median of. (These are scipy interp1D objects)
    :param bin_max: The value to extract the median out to. Defined as the maximum value of the largest profile.
    :param profile_division: The number of slices to divide the semi-major axis into
    """

    # Sample 100 times along the x-axis
    med_sma = arange(0, bin_max, bin_max / profile_division)

    med_intens, upper_sigma_1sig, lower_sigma_1sig = [], [], []

    for x_slice in med_sma:
        slice_values = []
        # Get all profile values at that slice
        for profile in pop:
            slice_value = profile(x_slice)
            # Check if value is nan or infinite. If that is the case, skip over it
            if slice_value in (nan, inf):
                continue

            slice_values.append(slice_value)

        # Sort the values (for upper and lower sigmas)
        slice_values = sort(slice_values)

        # Take the median value and append it to the median list.
        if ignore_nans:
            median_value = nanmedian(slice_values)
        else:
            median_value = median(slice_values)
        med_intens.append(median_value)

    # Return the median_sma values, and the median
    return med_sma, interp1d(med_sma, med_intens)


def bootstrap_uncertainty(pop, bin_max, iterations=101):
    """
    :param pop: The list of profiles in the bin. (These are scipy interp1D objects)
    :param bin_max: The value to extract medians out to. Defined as the maximum value of the largest profile.
    :param iterations: The number of bootstrap populations to make.
    """

    # Prepare a list of bootstrapped medians
    bootstrap_medians = []

    # Make our bootstrap populations. For each population, make a median
    for n in range(iterations):
        bootstrap_pop = choice(pop, size=len(pop), replace=True)
        bootstrap_medians.append(get_median(bootstrap_pop, bin_max)[1])

    bootstrap_sma = arange(0, bin_max, bin_max / 100)
    err_upper, err_lower = [], []

    lower_sigma_1sig, lower_sigma_2sig, lower_sigma_3sig = [], [], []
    upper_sigma_1sig, upper_sigma_2sig, upper_sigma_3sig = [], [], []
    upper_sigma_5sig, lower_sigma_5sig = [], []

    # Rerun the median method, but this time use the medians we generated from the bootstrapping
    for x in bootstrap_sma:
        slice_values = []
        for median in bootstrap_medians:
            slice_values.append(median(x))
        slice_values.sort()

        central_value = slice_values[int(floor(iterations / 2))]

        lower_index_1sig, upper_index_1sig = int(len(slice_values) * 0.159), int(len(slice_values) * 0.841)

        lower_index_2sig, upper_index_2sig = int(len(slice_values) * 0.023), int(len(slice_values) * 0.977)
        lower_index_3sig, upper_index_3sig = int(len(slice_values) * 0.002), int(len(slice_values) * 0.998)

        lower_sigma_1sig.append(slice_values[lower_index_1sig])
        upper_sigma_1sig.append(slice_values[upper_index_1sig])

        lower_sigma_2sig.append(slice_values[lower_index_2sig])
        upper_sigma_2sig.append(slice_values[upper_index_2sig])

        lower_sigma_3sig.append(slice_values[lower_index_3sig])
        upper_sigma_3sig.append(slice_values[upper_index_3sig])

        lower_sigma_5sig.append(slice_values[0])
        upper_sigma_5sig.append(slice_values[len(slice_values) - 1])

    # Return errors (for now as two arrays that can be plotted using the arange and bin max)
    return bootstrap_sma, \
        interp1d(bootstrap_sma, lower_sigma_1sig), interp1d(bootstrap_sma, upper_sigma_1sig), \
        interp1d(bootstrap_sma, lower_sigma_2sig), interp1d(bootstrap_sma, upper_sigma_2sig),  \
        interp1d(bootstrap_sma, lower_sigma_3sig), interp1d(bootstrap_sma, upper_sigma_3sig), \
        interp1d(bootstrap_sma, lower_sigma_5sig), interp1d(bootstrap_sma, upper_sigma_5sig)


def save_medians(median_data, bootstrap_data=None, output_filename="medians.fits"):
    median_sma, median_interp = median_data
    median_intens = median_interp(median_sma)

    out_hdulist = fits.HDUList()

    t = Table([median_sma, median_intens], names=["SMA", "INTENS"])
    out_hdulist.append(fits.BinTableHDU(t))
    if bootstrap_data is not None:
        b_sma, b_1sig_l, b_1sig_u, b_2sig_l, b_2sig_u, b_3sig_l, b_3sig_u, b_5sig_l, b_5sig_u = bootstrap_data
        # Append Lower Bootstrap Value
        t = Table([b_sma, b_1sig_l(b_sma), b_2sig_l(b_sma), b_3sig_l(b_sma), b_5sig_l(b_sma)],
                  names=["SMA", "INTENS_1SIG", "INTENS_2SIG", "INTENS_3SIG", "INTENS_5SIG"])
        out_hdulist.append(fits.BinTableHDU(t))
        # Append Upper Bootstrap Value
        t = Table([b_sma, b_1sig_u(b_sma), b_2sig_u(b_sma), b_3sig_u(b_sma), b_5sig_u(b_sma)],
                  names=["SMA", "INTENS_1SIG", "INTENS_2SIG", "INTENS_3SIG", "INTENS_5SIG"])
        out_hdulist.append(fits.BinTableHDU(t))

    out_hdulist.writeto(output_filename, overwrite=True)


def index_format(in_dir, x_bins, y_bins, out_dir="dir_copy/", indices=(1, 2), method='duplicate'):
    """
    Duplicate a directory but with the files in index format.
    :param in_dir: Input directory of median objects
    :param out_dir: Only needed if running in duplicate mode. Duplicates files in new directory
    :param x_bins: x-axis bin values (usually grabbed from config file)
    :param y_bins: y-axis bin values
    :param indices: When splitting the file, the indices correspond to x and y parameter locations.
        For example, if your filename is: bin_9.2-9.6_0.3-0.5_0.0-0.5_bare.fits
        and you want the x parameter to be 9.2-9.6 and the y parameter to be 0.3-0.5, you would
        set:
            indices=[1,2]
    :param method: duplicate directory or rename files
        'duplicate' : generate a new directory
        'rename' : Rename the files in input directory
    :return:
    """
    subdirs = os.listdir(in_dir)

    if method == 'duplicate' and not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for subdir in subdirs:

        if os.path.isfile(in_dir + subdir):
            continue

        subdir += "/"
        files = os.listdir(in_dir + subdir)

        if method == 'duplicate' and not os.path.isdir(out_dir + subdir):
            os.mkdir(out_dir + subdir)

        for f in files:
            splits = f.split("_")
            x_split = splits[indices[0]].split("-")
            x_val = (float(x_split[0]) + float(x_split[1])) / 2
            y_split = splits[indices[1]].split("-")
            y_val = (float(y_split[0]) + float(y_split[1])) / 2

            x_index, y_index = tbridge.bin_index(x_val, x_bins), tbridge.bin_index(y_val, y_bins)

            # print(f, "|", x_val, x_index, y_val, y_index)
            new_filename = str(x_index) + "_" + str(y_index) + "_" + f.split("_")[-1]
            if method == 'duplicate':
                shutil.copyfile(in_dir + subdir + f, out_dir + subdir + new_filename)
            elif method == 'rename':
                os.rename(in_dir + subdir + f, in_dir + subdir + new_filename)


def __median_processing(full_filename, config, out_dir="", subdir="", iterations=101,
                        x_key="REDSHIFT_BINS", y_key="MASS_BINS", x_index=1, y_index=0, index_format=True):
    """
    Run through the median processing on a given filename.
    :param full_filename:
    :param out_dir:
    :param subdir:
    :param index_format: Save files in an index format (x_y_suffix.fits)
    :return:
    """

    params = tbridge.params_from_filename(full_filename)

    bins_x, bins_y = config[x_key], config[y_key]

    x = tbridge.bin_index(params[x_index], bins_x)
    y = tbridge.bin_index(params[y_index], bins_y)

    # filename = full_filename.split("/")[len(full_filename.split("/")) - 1]

    # filename_suffix = full_filename.split("_")[len(full_filename.split("_")) - 1].split(".")[0] + "_median.fits"
    # filename = str(x) + "_" + str(y) + "_" + filename_suffix

    filename = full_filename.split("/")[-1]

    prof_list = tbridge.tables_from_file(full_filename)
    bin_max_value = bin_max(prof_list)
    prof_list = as_interpolations(prof_list)

    med_data = tbridge.get_median(prof_list, bin_max_value)
    bootstrap_data = tbridge.bootstrap_uncertainty(prof_list, bin_max_value, iterations=iterations)
    tbridge.save_medians(med_data, bootstrap_data, output_filename=out_dir + subdir + filename)


def median_pipeline(in_dir, config=tbridge.default_config_params(), multiprocess=False, cores=1, iterations=101):

    out_dir = in_dir[:len(in_dir) - 1] + "_medians/"

    subdirs = os.listdir(in_dir)
    tbridge.generate_file_structure(out_dir, subdirs)

    for subdir in subdirs[:]:
        if os.path.isfile(in_dir + subdir):
            continue
        subdir = subdir + "/"

        bins = [str(b) for b in Path(in_dir + subdir).rglob('*.fits')]

        print(subdir, " | Bins:", len(bins))

        if multiprocess:
            pool = mp.Pool(processes=cores)
            results = [pool.apply_async(__median_processing, (b, config, out_dir, subdir, iterations)) for b in bins]
            [res.get() for res in results]
        else:
            for b in bins:
                __median_processing(b, config, out_dir, subdir, iterations)


def load_median_info(filename):
    """
    Load a complete set of median info (TBriDGE output) in as a dict object.
    :param filename:
    :return:
    """
    median_data = {}

    hdul = fits.open(filename)
    med, l, u = Table.read(hdul[1]), Table.read(hdul[2]), Table.read(hdul[3])

    median_data["MED_SMA"] = med["SMA"]
    median_data["MED_INTENS"] = med["INTENS"]

    median_data["MED_ADJ"] = tbridge.adjust_profile(med["SMA"], med["INTENS"])

    median_data["L_1SIG"] = (l["SMA"], l["INTENS_1SIG"])
    median_data["U_1SIG"] = (u["SMA"], u["INTENS_1SIG"])
    median_data["L_1SIG_ADJ"] = tbridge.adjust_profile(l["SMA"], l["INTENS_1SIG"])
    median_data["U_1SIG_ADJ"] = tbridge.adjust_profile(u["SMA"], u["INTENS_1SIG"])

    median_data["L_2SIG"] = (l["SMA"], l["INTENS_2SIG"])
    median_data["U_2SIG"] = (u["SMA"], u["INTENS_2SIG"])
    median_data["L_2SIG_ADJ"] = tbridge.adjust_profile(l["SMA"], l["INTENS_2SIG"])
    median_data["U_2SIG_ADJ"] = tbridge.adjust_profile(u["SMA"], u["INTENS_2SIG"])

    median_data["L_3SIG"] = (l["SMA"], l["INTENS_3SIG"])
    median_data["U_3SIG"] = (u["SMA"], u["INTENS_3SIG"])
    median_data["L_3SIG_ADJ"] = tbridge.adjust_profile(l["SMA"], l["INTENS_3SIG"])
    median_data["U_3SIG_ADJ"] = tbridge.adjust_profile(u["SMA"], u["INTENS_3SIG"])

    median_data["L_5SIG"] = (l["SMA"], l["INTENS_5SIG"])
    median_data["U_5SIG"] = (u["SMA"], u["INTENS_5SIG"])
    median_data["L_5SIG_ADJ"] = tbridge.adjust_profile(l["SMA"], l["INTENS_5SIG"])
    median_data["U_5SIG_ADJ"] = tbridge.adjust_profile(u["SMA"], u["INTENS_5SIG"])

    return median_data
