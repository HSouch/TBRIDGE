import os, time, warnings, argparse
from pathlib import Path
import multiprocessing as mp
from astropy.io import fits
from astropy.modeling.models import Sersic2D
from astropy.wcs import wcs
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.table import Table
from numpy import unravel_index, argmax, mgrid, floor, ceil, copy, str
from numpy import array, ndarray, log, exp, isnan, sum, sqrt
from numpy import arange, median, mean, max, copy, pi, round
from numpy.random import choice, randint, uniform
from photutils import detect_threshold, detect_sources, deblend_sources, data_properties, make_source_mask
from photutils.datasets import make_noise_image

from scipy.stats import gaussian_kde
from scipy.special import gamma
from scipy.signal import convolve2d


def process(image_band="i"):
    """ Prepare all bins for processing (main code wrapper) """

    # Read HST-ZEST catalog into memory
    catalog = Table.read(catalog_filename, format="fits")

    mags = array(catalog[mag_cn])
    r50s = array(catalog[r50_cn])
    ns = array(catalog[n_cn])
    ellips = array(catalog[ellip_cn])
    masses = array(catalog[mass_cn])
    sfprobs = array(catalog[sfprob_cn])
    redshifts = array(catalog[redshift_cn])

    bins, empty_bins = [], []

    # For each bin:
    for low_sfprob in sfprob_bins[:]:
        for low_redshift in redshift_bins[:]:
            for low_mass in mass_bins[:]:
                this_bin = []
                # Check all objects and catalog to see if any fit in this bin
                for n in range(0, len(catalog)):
                    if low_mass < masses[n] < low_mass + mass_step \
                            and low_redshift < redshifts[n] < low_redshift + redshift_step \
                            and low_sfprob < sfprobs[n] < low_sfprob + sfprob_step:
                        # Add index to bin if no values are flagged (set to -99)
                        if r50s[n] > 0 and ns[n] > 0 and ellips[n] > 0 and mags[n] > 0:
                            this_bin.append(n)
                bin_r50s, bin_ns, bin_ellips, bin_mags = [], [], [], []

                # Don't bother with this bin if it is less than the preset number threshold.
                if len(this_bin) < number_threshold:
                    empty_bins.append([low_mass, low_redshift, low_sfprob])
                    continue
                for index in this_bin:
                    bin_r50s.append(r50s[index])
                    bin_mags.append(mags[index])
                    bin_ns.append(ns[index])
                    bin_ellips.append(ellips[index])
                bins.append({"params": (bin_mags, bin_r50s, bin_ns, bin_ellips), "low_sfprob": low_sfprob,
                             "low_redshift": low_redshift, "low_mass": low_mass, "mass_step": mass_step,
                             "sfprob_step": sfprob_step, "redshift_step": redshift_step})

    print(len(bins), "bins to process that have a suitable number of candidates.", len(empty_bins), "have no objects!")

    # Process all the bins. Set processes in multithreading in accordance to test mode being on or not.
    # If in test mode, set to a laptop friendly number of cores. Otherwise set to AceNet node size (32 cores)
    if test_mode:
        print("model_matrix is running in test mode with", test_cores, "cores.")
        pool = mp.Pool(processes=test_cores)
    else:
        print("model_matrix is running in full mode with", main_cores, "cores.")
        pool = mp.Pool(processes=main_cores)

    if test_mode:
        bins = bins[:bin_max]

    # Only run on this number of bins (run on everything if not in test mode obviously)
    results = [pool.apply_async(process_bin, (object_bin, clauds_mags)) for object_bin in bins]
    [res.get() for res in results]


def process_bin(object_bin, clauds_mags):
    """ Run simulation code on a single bin """
    # Load in information from "bin" dict object
    bin_params = object_bin["params"]
    low_sfprob, low_redshift, low_mass = object_bin["low_sfprob"], object_bin["low_redshift"], object_bin["low_mass"]

    # Set cutout parameters
    x_halfwidth, y_halfwidth = ceil(global_cutout_size / 2), ceil(global_cutout_size / 2)
    x, y = mgrid[:global_cutout_size, :global_cutout_size]

    # Prepare output lists
    bare_models, psfs, bg_added_models = [], [], []
    out_tables_bare, out_tables_bgadded, out_tables_noisy, out_tables_localsub, out_tables_psf = [], [], [], [], []
    header_info = []

    # Get all possible images to pull from
    img_filename_list = get_image_filenames(image_directory, image_band=band)
    print(len(img_filename_list), "images to choose from.")

    # Run model extraction on n different images
    for image_index in range(processing_runs):
        # Get all available object parameters. Create PDF and resample parameters
        mags, r50s, ns, ellips = bin_params[0], bin_params[1], bin_params[2], bin_params[3]
        object_resample = gaussian_kde((mags, r50s, ns, ellips)).resample(1000)
        mags_resample, r50s_resample = object_resample[0], object_resample[1]
        ns_resample, ellips_resample = object_resample[2], object_resample[3]

        # Get CLAUDS mags instead of ZEST mags for resampling
        mags_resample = gaussian_kde(clauds_mags).resample(1000)[0]

        # Select an image based on a random selection (either random or through the coverage table)

        if use_coverage_map:
            coverage_table = Table.read("coverage_cosmos.fits", format="fits")
            img_filename = random_selection(coverage_table, ra_min=148.75, ra_max=151.5,
                                            dec_min=0.75, dec_max=3.75, band=band)
        elif use_same_bg:
            print("Using same bg")
            img_filename = premade_image_list[image_index]
        elif global_use_loc_catalogue:
            img_filename = loc_images[image_index]
        else:
            img_filename = choice(img_filename_list)

        # img_filename_no_path = img_filename.split("/")[len(img_filename.split("/")) - 1]

        try:
            img_hdulist = fits.open(img_filename)
        except:
            print("Issue opening image:", img_filename)
            continue

        psf_hdulist = fits.open(psf_filename)
        try:
            image = img_hdulist[1].data
        except IndexError:
            try:
                image = img_hdulist[0].data
            except:
                print("Cannot Open image")
                continue
        except:
            print("Cannot Open image")
            continue
        try:
            img_wcs = get_wcs(img_filename)
        except:
            print("Issue opening wcs:", img_filename)
            continue

        xs = arange(x_halfwidth + 1, image.shape[0] - x_halfwidth, image.shape[0] / 2)
        ys = arange(x_halfwidth + 1, image.shape[1] - y_halfwidth, image.shape[1] / 2)

        def generate_model(x_pos, y_pos):
            """ Make the necessary model images at a pre-defined position in the image"""

            def equal_sizes(arr_1, arr_2):
                """ Simple way to ensure cutouts are of the same size."""
                return arr_1[0:arr_2.shape[0], 0:arr_2.shape[1]], arr_2[0:arr_1.shape[0], 0:arr_1.shape[1]]

            # Get sampled parameters. Skip this attempt if it fails more than 10000 times.
            fails, clean = 0, False
            this_r50, this_n, this_ellip, this_mag, this_r50_pix = None, None, None, None, None
            while not clean:
                if fails > 10000:
                    break
                obj_choice = randint(0, len(mags_resample))
                this_mag = mags_resample[obj_choice]
                this_r50 = r50s_resample[obj_choice]  # This is in arcseconds
                this_r50_pix = this_r50 / arcconv  # This is in pixels
                this_n = ns_resample[obj_choice]
                this_ellip = ellips_resample[obj_choice]

                # Get the intensity at the half-light radius. Divide by the exposure time
                i_r50 = i_at_r50(mag=this_mag, n=this_n, r_50=this_r50)
                clean = not (i_r50 > 100) and not (isnan(i_r50)) and (this_n > 0.65) and (0 < this_ellip < 1) \
                        and (this_r50_pix > 0)
                if not clean:
                    fails += 1
            if fails > 10000:
                return None

            # Generate Sersic Model (unconvolved)
            sersic_model = Sersic2D(amplitude=i_r50, r_eff=this_r50_pix, n=this_n,
                                    x_0=global_cutout_size / 2, y_0=global_cutout_size / 2,
                                    ellip=this_ellip, theta=uniform(0, 2 * pi))
            z = sersic_model(x, y)

            # Get closest psf, convolve it with the model, and append the model
            coord = img_wcs.wcs_pix2world(x_pos, y_pos, 0)
            ra, dec = coord[0], coord[1]
            psf = get_closest_psf(psf_hdulist, ra, dec).data

            z_conv = convolve2d(z, psf, mode="same")
            z_conv *= (sum(z) / sum(z_conv))

            # Get an image from a list of cutouts if specified
            if global_use_cutout_bgs:
                image_cutout = select_image(cutout_filename)
                image_cutout, z_conv = equal_sizes(image_cutout, z_conv)
            # If not, get the image from a selection
            else:
                x_min, x_max = x_pos - x_halfwidth, x_pos + x_halfwidth
                y_min, y_max = y_pos - y_halfwidth, y_pos + y_halfwidth
                try:
                    image_cutout = image[int(x_min):int(x_max) - 1, int(y_min):int(y_max) - 1]
                except IndexError:
                    return None

                # If the cutout is too close to an edge scrap it
                if image_cutout.shape != z_conv.shape:
                    return None

            bga = image_cutout + z_conv

            try:
                masked_bga = mask_cutout(bga)[0]
            except AttributeError:
                masked_bga = bga

            bare_models.append(z_conv)
            psfs.append(psf)
            bg_added_models.append(masked_bga)

            header_info.append({"img_filename": img_filename,
                                "r50": this_r50,
                                "i_r50": i_r50,
                                "r50_pix": this_r50_pix,
                                "mag": this_mag,
                                "n": this_n,
                                "ellip": this_ellip})

        if global_use_loc_catalogue:
            x_loc, y_loc = img_wcs.wcs_world2pix(loc_ras[image_index], loc_decs[image_index], 0)
            x_loc, y_loc = int(x_loc), int(y_loc)
            generate_model(x_loc, y_loc)
        elif global_use_cutout_bgs:
            generate_model(0, 0)
        else:
            for x_index in xs:
                for y_index in ys:
                    generate_model(x_index, y_index)

        psf_hdulist.close()
        img_hdulist.close()

    # #######################################################################################
    # Extract profiles. Only save the ones that are both extracted successfully
    # #######################################################################################
    successful, failed = 0, 0
    print(len(bare_models) * 3, "profiles to extract in", low_mass, low_redshift, low_sfprob)

    for n in range(0, len(bare_models)):
        if test_mode:
            print(n)
        isolist_bare = isophote_fitting(bare_models[n])
        isolist_bgadded = isophote_fitting(bg_added_models[n])
        isolist_psf = isophote_fitting(psfs[n])

        # bg_est, bg_rms = estimate_background(bg_added_models[n])
        noise_image = make_noise_image((global_cutout_size, global_cutout_size), mean=0, stddev=0.025)

        noisy_model = noise_image + bare_models[n]
        isolist_noisy = isophote_fitting(noisy_model)

        # If successful fit, save output
        if len(isolist_bare) > 0 and len(isolist_bgadded) > 0 and len(isolist_noisy) > 0:
            successful += 1

            isotable_bare = Table(isolist_bare.to_table())
            isotable_bgadded = Table(isolist_bgadded.to_table())
            isotable_noisy = Table(isolist_noisy.to_table())

            # Subtract Local Background
            try:
                localbg_source_mask = make_source_mask(bg_added_models[n], nsigma=2, npixels=5)
            except TypeError:
                localbg_source_mask = make_source_mask(bg_added_models[n], snr=2, npixels=5)

            bg_mean, bg_median, bg_std = sigma_clipped_stats(bg_added_models[n], mask=localbg_source_mask)

            isotable_localsub = Table()
            for col in isotable_bgadded.colnames:
                isotable_localsub[col] = copy(isotable_bgadded[col])
            isotable_localsub["intens"] = ((isotable_localsub["intens"]) - bg_median)

            # These are the columns to write
            valid_colnames = ["sma", "intens", "intens_err", "ellipticity", "ellipticity_err", "pa", "pa_err"]

            # Create and populate header
            hdr = fits.Header()
            hdr["BG_EST"], hdr["BG_STD"], hdr["BAND"] = bg_median, bg_std, band
            obj_info = header_info[n]

            for key in obj_info:
                hdr[key] = obj_info[key]

            if len(isolist_psf) > 0:
                isotable_psf = Table(isolist_psf.to_table())
                out_tables_psf.append(fits.BinTableHDU(Table([isotable_psf[col] for col in valid_colnames],
                                                             names=valid_colnames), header=hdr))

            out_tables_bare.append(fits.BinTableHDU(Table([isotable_bare[col] for col in valid_colnames],
                                                          names=valid_colnames), header=hdr))
            out_tables_bgadded.append(fits.BinTableHDU(Table([isotable_bgadded[col] for col in valid_colnames],
                                                             names=valid_colnames), header=hdr))
            out_tables_noisy.append(fits.BinTableHDU(Table([isotable_noisy[col] for col in valid_colnames],
                                                           names=valid_colnames), header=hdr))
            out_tables_localsub.append(fits.BinTableHDU(Table([isotable_localsub[col] for col in valid_colnames],
                                                           names=valid_colnames), header=hdr))
        else:
            failed += 1

    primary_header = fits.Header()
    primary_header["success"] = successful
    primary_header["fail"] = failed

    # Save files with unique and reverse-engineerable filename (for plotting and later processing)
    print("Saving", low_mass, low_redshift, low_sfprob, "for", len(out_tables_bare), "objects.")

    output_filename = "bin_" + str(low_mass) + "-" + str(round(low_mass + mass_step, 2)) + "_"
    output_filename += str(low_redshift) + "-" + str(round(low_redshift + redshift_step, 2)) + "_"
    output_filename += str(low_sfprob) + "-" + str(round(low_sfprob + sfprob_step, 2))

    bare_profile_filename = output_filename + "_bareprofile.fits"
    bgadded_profile_filename = output_filename + "_bgaddedprofile.fits"
    noisy_profile_filename = output_filename + "_noisyprofile.fits"
    psf_profile_filename = output_filename + "_psfprofile.fits"
    localsub_profile_filename = output_filename + "_localsubprofile.fits"

    if len(out_tables_bare) > 0 and len(out_tables_bgadded) > 0 and len(out_tables_noisy) > 0:
        bare_profiles = fits.HDUList()
        bare_profiles.append(fits.PrimaryHDU(header=primary_header))
        for hdu in out_tables_bare:
            bare_profiles.append(hdu)
        bare_profiles.writeto(bare_profile_outdir + bare_profile_filename, overwrite=True)
        bare_profiles.close()

        bgadded_profiles = fits.HDUList()
        bgadded_profiles.append(fits.PrimaryHDU(header=primary_header))
        for hdu in out_tables_bgadded:
            bgadded_profiles.append(hdu)
        bgadded_profiles.writeto(bgadded_profile_outdir + bgadded_profile_filename, overwrite=True)
        bgadded_profiles.close()

        noisy_profiles = fits.HDUList()
        noisy_profiles.append(fits.PrimaryHDU(header=primary_header))
        for hdu in out_tables_noisy:
            noisy_profiles.append(hdu)
        noisy_profiles.writeto(noisy_outdir + noisy_profile_filename, overwrite=True)
        noisy_profiles.close()

        localsub_profiles = fits.HDUList()
        localsub_profiles.append(fits.PrimaryHDU(header=primary_header))
        for hdu in out_tables_localsub:
            localsub_profiles.append(hdu)
        localsub_profiles.writeto(localsub_outdir + localsub_profile_filename, overwrite=True)
        localsub_profiles.close()

    if len(out_tables_psf) > 0:
        psf_profiles = fits.HDUList()
        psf_profiles.append(fits.PrimaryHDU(header=primary_header))
        for hdu in out_tables_psf:
            psf_profiles.append(hdu)
        psf_profiles.writeto(psf_outdir + psf_profile_filename, overwrite=True)
