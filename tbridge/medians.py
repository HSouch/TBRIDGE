from numpy import arange, nan, inf, sort, median, floor, max
from numpy.random import choice
from scipy.interpolate import interp1d

import multiprocessing as mp

from astropy.io import fits
from astropy.table import Table

import tbridge
import os
from pathlib import Path

def as_interpolations(profile_list, fill_value_type='min', x_key="sma", y_key="intens"):

    interps = []

    for prof in profile_list:
        sma, intens = prof[x_key], prof[y_key]
        interp = interp1d(sma, intens, bounds_error=False, fill_value=0)
        interps.append(interp)

    return interps


def bin_max(profile_list, key="sma"):
    max_val = -999

    for prof in profile_list:
        arr_max = max(prof[key])
        max_val = arr_max if arr_max > max_val else max_val

    return max_val


def get_median(pop, bin_max):
    """
    Obtain the median profile for a bin of profiles. Takes "slices" along the x-axis, obtaining all profile values at
    that slice, and populates a list of median values. Also obtains the upper and lower sigma values.

    pop: The list of profiles to get the median of. (These are scipy interp1D objects)
    bin_max: The value to extract the median out to. Defined as the maximum value of the largest profile.
    """

    # Sample 100 times along the x-axis
    med_sma = arange(0, bin_max, bin_max / 100)

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


def __median_processing(full_filename, out_dir="", subdir=""):
    """
    Run through the median processing on a given filename.
    :param full_filename:
    :param out_dir:
    :param subdir:
    :return:
    """
    filename = full_filename.split("/")[len(full_filename.split("/")) - 1]
    print(filename, subdir)
    prof_list = tbridge.tables_from_file(full_filename)

    bin_max_value = bin_max(prof_list)

    prof_list = as_interpolations(prof_list)

    med_data = tbridge.get_median(prof_list, bin_max_value)
    bootstrap_data = tbridge.bootstrap_uncertainty(prof_list, bin_max_value, iterations=5)
    tbridge.save_medians(med_data, bootstrap_data, output_filename=out_dir + subdir + filename)


def median_pipeline(in_dir, multiprocess=False, cores=1):

    out_dir = in_dir[:len(in_dir) - 1] + "_medians/"

    subdirs = os.listdir(in_dir)
    tbridge.generate_file_structure(out_dir, subdirs)

    for subdir in subdirs:
        subdir = subdir + "/"

        bins = [str(b) for b in Path(in_dir + subdir).rglob('*.fits')]

        if multiprocess:
            pool = mp.Pool(processes=cores)
            results = [pool.apply_async(__median_processing, (b, out_dir, subdir)) for b in bins]
            [res.get() for res in results]
        else:
            for b in bins:
                __median_processing(b, out_dir, subdir)