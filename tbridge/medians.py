from numpy import arange, nan, inf, sort, median, floor, max, nanmedian, transpose, asarray, std
from numpy.random import choice, randint
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
            continue
    return interps


def bin_max(profile_list, key="sma"):
    max_val = -999

    for prof in profile_list:
        arr_max = max(prof[key])
        max_val = arr_max if arr_max > max_val else max_val

    return max_val


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

    def generate_file_structure(out_dir, subdirs):
        """ Generate a filestructure with N subdirs within the top directory out_dir"""
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        for subdir in subdirs:
            if not os.path.isdir(out_dir + subdir + "/"):
                os.mkdir(out_dir + subdir + "/")

    subdirs = os.listdir(in_dir)
    generate_file_structure(out_dir, subdirs)

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


def bin_max(profiles, key="sma"):
    bin_max = -999
    for prof in profiles:
        try:
            prof_max = prof[key][-1]
            if prof_max > bin_max:
                bin_max = prof_max
        except Exception as error:
            continue
    return bin_max


def normalize_bin(profiles, bin_max, step=0.5, key="sma"):
    new_sma = arange(0, bin_max + step, step)
    normalized_profiles = [prof(new_sma) for prof in profiles]

    return new_sma, asarray(normalized_profiles)


def normalized_median(normalized_profiles, handle_nans=True):
    if handle_nans:
        median_prof = nanmedian(normalized_profiles, axis=0)
    else:
        median_prof = median(normalized_profiles, axis=0)
    # print(median_prof.shape)
    return median_prof


def normalized_bootstrap(normalized_profiles, iterations=1001, gaussian=True, handle_nans=True):
    true_median = normalized_median(normalized_profiles)

    medians = []
    bin_size = len(normalized_profiles)
    for i in range(iterations):
        pop = normalized_profiles[randint(0, bin_size, size=bin_size)]
        medians.append(normalized_median(pop, handle_nans=handle_nans))

    medians = asarray(medians)

    # If we can assume a Gaussian distribution, this part is easy
    if gaussian:
        slices = std(transpose(medians), axis=1)

        lower_1sig = true_median - slices
        upper_1sig = true_median + slices

        lower_2sig = true_median - 2 * slices
        upper_2sig = true_median + 2 * slices

        lower_3sig = true_median - 3 * slices
        upper_3sig = true_median + 3 * slices

        lower_5sig = true_median - 5 * slices
        upper_5sig = true_median + 5 * slices

        return {"L_1SIG": lower_1sig, "L_2SIG": lower_2sig, "L_3SIG": lower_3sig, "L_5SIG": lower_5sig,
                "U_1SIG": upper_1sig, "U_2SIG": upper_2sig, "U_3SIG": upper_3sig, "U_5SIG": upper_5sig}

    else:
        slices = sort(transpose(medians))

        lower_index_1sig, upper_index_1sig = int(iterations * 0.159), int(iterations * 0.841)
        lower_index_2sig, upper_index_2sig = int(iterations * 0.023), int(iterations * 0.977)
        lower_index_3sig, upper_index_3sig = int(iterations * 0.002), int(iterations * 0.998)

        lower_1sig = [median_slice[lower_index_1sig] for median_slice in slices]
        upper_1sig = [median_slice[upper_index_1sig] for median_slice in slices]

        lower_2sig = [median_slice[lower_index_2sig] for median_slice in slices]
        upper_2sig = [median_slice[upper_index_2sig] for median_slice in slices]

        lower_3sig = [median_slice[lower_index_3sig] for median_slice in slices]
        upper_3sig = [median_slice[upper_index_3sig] for median_slice in slices]

        lower_5sig = [median_slice[0] for median_slice in slices]
        upper_5sig = [median_slice[len(median_slice) - 1] for median_slice in slices]

        return {"L_1SIG": lower_1sig, "L_2SIG": lower_2sig, "L_3SIG": lower_3sig, "L_5SIG": lower_5sig,
                "U_1SIG": upper_1sig, "U_2SIG": upper_2sig, "U_3SIG": upper_3sig, "U_5SIG": upper_5sig}


def median_pipeline(full_filename, output_filename=None, step=0.5):
    profiles = tbridge.load_profile_set(full_filename)

    bin_max_value = bin_max(profiles)
    median_sma, normalized_profiles = normalize_bin(tbridge.as_interpolations(profiles), bin_max_value,
                                                    step=step)

    median_profile = normalized_median(normalized_profiles)
    b_dict = normalized_bootstrap(normalized_profiles)

    if output_filename is not None:
        out_hdulist = fits.HDUList()

        med_table = Table([median_sma, median_profile], names=["SMA", "INTENS"])

        b_lower_table = Table([median_sma, b_dict["L_1SIG"],
                               b_dict["L_2SIG"], b_dict["L_3SIG"], b_dict["L_5SIG"]],
                              names=["SMA", "INTENS_1SIG", "INTENS_2SIG", "INTENS_3SIG", "INTENS_5SIG"])

        b_upper_table = Table([median_sma, b_dict["U_1SIG"], b_dict["U_2SIG"],
                               b_dict["U_3SIG"], b_dict["U_5SIG"]],
                              names=["SMA", "INTENS_1SIG", "INTENS_2SIG", "INTENS_3SIG", "INTENS_5SIG"])

        out_hdulist.append(fits.BinTableHDU(med_table))
        out_hdulist.append(fits.BinTableHDU(b_lower_table))
        out_hdulist.append(fits.BinTableHDU(b_upper_table))

        out_hdulist.writeto(output_filename, overwrite=True)

        return b_dict.update({"MEDIAN": med_table, "SMA": median_sma})
