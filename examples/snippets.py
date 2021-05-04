"""
Contains some out of date code and other snippets that might be useful in the future
"""


# test = tbridge.extract_profiles([masked_cutouts[:]], config=config_values, progress_bar=True, isophote_testing=False)[0]
#
# for n in test:
#     print(test)
#
# print(len(masked_cutouts), len(test))
#
# plotter.single_bin_plot([test], colours=["Red"], medians=True, ind_profile_alpha=0.2)
# plotter.view_cutouts(masked_cutouts[], log_scale=False)
# plotter.matrix_plot("out_i_medians/", x_bins=config_values["MASS_BINS"], y_bins=config_values["REDSHIFT_BINS"])

# topdir = "tbridge_medians/"
# for directory in os.listdir(topdir):
#     print(directory)
#     tbridge.index_format(topdir + directory + "/", x_bins=config_values["MASS_BINS"], y_bins=config_values["REDSHIFT_BINS"],
#                          method='duplicate', out_dir="reworked/" + directory + "/")


# bare_profiles = tbridge.tables_from_file("out_i/bare_profiles/bin_9.6-10.0_0.1-0.3_0.0-0.5_bare.fits")
# bgadded_profiles = tbridge.tables_from_file("out_i/bgadded_profiles/bin_9.6-10.0_0.1-0.3_0.0-0.5_bgadded.fits")
# noisy_profiles = tbridge.tables_from_file("out_i/noisy_profiles/bin_9.6-10.0_0.1-0.3_0.0-0.5_noisy.fits")
# bgsub_profiles = tbridge.tables_from_file("out_i/bgsub_profiles/bin_9.6-10.0_0.1-0.3_0.0-0.5_bgsub.fits")
#
# plotter.single_bin_plot([bare_profiles, bgadded_profiles, noisy_profiles, bgsub_profiles],
#                         colours=["red", "violet", "green", "blue"], medians=True, ind_profile_alpha=0.,
#                         xlabel="SMA [pix]", ylabel="Surface Brightness")



# def _process_bin_old(b, config_values, separate_mags=None, provided_bgs=None, progress_bar=False, multiprocess=False):
#     """
#     Process a single bin of galaxies. (Tuned for pipeline usage but can be used on an individual basis.
#     :param b: Bin to obtain full profiles from.
#     :param config_values: Values from properly loaded configuration file.
#     :param separate_mags: Optional array of magnitudes.
#     :param provided_bgs: Array of provided backgrounds
#     :param progress_bar: Use a TQDM progress bar (note with multithreading this might get funky).
#     :param multiprocess: Run in multiprocessing mode.
#         This means both the model creation AND the
#     :return:
#     """
#
#     t_start = time.time()
#     verbose = config_values["VERBOSE"]
#
#     # Load in information
#     keys, columns = b.return_columns()
#     mags, r50s, ns, ellips = tbridge.pdf_resample(tbridge.structural_parameters(keys, columns), resample_size=1000)
#     if separate_mags is not None:
#         mags = tbridge.pdf_resample(separate_mags, resample_size=len(r50s))[0]
#
#     if multiprocess:
#         if verbose:
#             print("Simulating", config_values["N_MODELS"], "models for: ", b.bin_params)
#
#         with mp.Pool(processes=config_values["CORES"]) as pool:
#             models = tbridge.simulate_sersic_models(mags, r50s, ns, ellips,
#                                                     config_values, n_models=config_values["N_MODELS"])
#
#             results = [pool.apply_async(_simulate_single_model, (models[i], config_values, provided_bgs))
#                        for i in range(0, len(models))]
#
#             model_list = []
#             for res in results:
#                 try:
#                     model_list.append(res.get(timeout=config_values["ALARM_TIME"] * 2))
#                 except TimeoutError:
#                     print("Simulation TimeoutError")
#                     continue
#
#             pool.terminate()
#
#         # Get all profile lists from our developed models.
#         if verbose:
#             print("Extracting Profiles for: ", b.bin_params)
#
#         with mp.Pool(processes=config_values["CORES"]) as pool:
#             results = [pool.apply_async(tbridge.extract_profiles_single_row,
#                                         (model_list[i][0], config_values, model_list[i][1]))
#                        for i in range(0, len(model_list))]
#
#             full_profile_list, timed_out_rows = [], []
#             for res in results:
#                 try:
#                     full_profile_list.append(res.get(timeout=config_values["ALARM_TIME"]))
#                 except TimeoutError:
#                     print("Extraction TimeoutError")
#                     continue
#
#             pool.terminate()
#
#         # If nothing worked just go to the next bin
#         if full_profile_list is None or len(full_profile_list) == 0:
#             return
#
#         # Trim all empty arrays from the profile list
#         profile_list = [row for row in full_profile_list if len(row[0]) > 0]
#
#         bg_info = []
#         for i in range(0, len(profile_list)):
#             # Every row is going to be a tuple with the list of model images, and the background info for that row.
#             row = profile_list[i]
#             bg_info.append(row[1])
#             profile_list[i] = row[0]
#
#         bg_info = transpose(bg_info)
#
#         # Reformat into a column-format
#         profile_list = _reformat_profile_list(profile_list)
#
#     # If not, simulate the bin serially
#     else:
#         if verbose:
#             print("Simulating Models for: ", b.bin_params)
#         # Generate all Sersic models
#         models = tbridge.simulate_sersic_models(mags, r50s, ns, ellips, config_values,
#                                                 n_models=config_values["N_MODELS"])
#         # Generate BG added models in accordance to whether a user has provided backgrounds or not
#         if provided_bgs is None:
#             bg_added_models, convolved_models = tbridge.add_to_locations_simple(models[:], config_values)
#             bg_added_models, bg_info = tbridge.mask_cutouts(bg_added_models)
#         else:
#             convolved_models = tbridge.convolve_models(models, config_values)
#             bg_added_models = tbridge.add_to_provided_backgrounds(convolved_models, provided_bgs)
#             bg_added_models, bg_info = tbridge.mask_cutouts(bg_added_models)
#
#         noisy_models = tbridge.add_to_noise(convolved_models)
#
#         if verbose:
#             print("Extracting Profiles for: ", b.bin_params)
#
#         profile_list = tbridge.extract_profiles((convolved_models, noisy_models, bg_added_models), config_values,
#                                                 progress_bar=progress_bar)
#
#     # Only save the profile in this bin if we have at least 1 set of profiles to save
#     if len(profile_list[0]) > 0:
#         if verbose:
#             print(len(profile_list[0]), "profiles extracted, wrapping up: ", b.bin_params)
#
#         # Estimate backgrounds and generate bg-subtracted profile list
#         backgrounds = bg_info[1]
#         bgsub_profiles = tbridge.subtract_backgrounds(profile_list[2], backgrounds)
#         profile_list.append(bgsub_profiles)
#
#         # Save profiles
#         tbridge.save_profiles(profile_list,
#                               bin_info=b.bin_params,
#                               out_dir=config_values["OUT_DIR"],
#                               keys=["bare", "noisy", "bgadded", "bgsub"],
#                               bg_info=bg_info)
#
#         unmasked_cutouts = array([row[2] for row in model_list])
#         full_cutouts = array([row[0][2] for row in model_list])
#
#         if config_values["SAVE_CUTOUTS"].lower() == 'stitch':
#             output_filename = config_values["OUT_DIR"] + tbridge.generate_file_prefix(b.bin_params) + "stitch.fits"
#             indices = choice(len(full_cutouts), size=int(len(full_cutouts) * config_values["CUTOUT_FRACTION"]),
#                              replace=False)
#             full_cutouts = full_cutouts[indices]
#             unmasked_cutouts = unmasked_cutouts[indices]
#             tbridge.cutout_stitch(unmasked_cutouts, masked_cutouts=full_cutouts, output_filename=output_filename)
#
#         if config_values["SAVE_CUTOUTS"].lower() == 'mosaic':
#             output_filename = config_values["OUT_DIR"] + tbridge.generate_file_prefix(b.bin_params) + ".png"
#             full_cutouts = full_cutouts[choice(len(full_cutouts),
#                                                size=int(len(full_cutouts) * config_values["CUTOUT_FRACTION"]),
#                                                replace=False)]
#             plotter.view_cutouts(full_cutouts, output=output_filename, log_scale=False)
#         if config_values["SAVE_CUTOUTS"].lower() == 'fits':
#             output_filename = config_values["OUT_DIR"] + tbridge.generate_file_prefix(b.bin_params) + ".png"
#             full_cutouts = full_cutouts[choice(len(full_cutouts),
#                                                size=int(len(full_cutouts) * config_values["CUTOUT_FRACTION"]),
#                                                replace=False)]
#             tbridge.save_cutouts(full_cutouts, output_filename=output_filename)
#
#     if verbose:
#         print("Finished", b.bin_params, "-- Time Taken:", round((time.time() - t_start) / 60, 2), "minutes.")
#         if multiprocess:
#             print()
#
#
# def _simulate_single_model(sersic_model, config_values, provided_bgs=None):
#     """
#
#     :param sersic_model:
#     :param config_values:
#     :param provided_bgs:
#     :return:
#     """
#     # Generate BG added models in accordance to whether a user has provided backgrounds or not
#     if provided_bgs is None:
#         bg_added_model, convolved_model = tbridge.add_to_locations_simple(sersic_model, config_values)
#     else:
#         convolved_model = tbridge.convolve_models(sersic_model, config_values)
#         bg_added_model = tbridge.add_to_provided_backgrounds(convolved_model, provided_bgs)
#
#     masked_model, mask_data = tbridge.mask_cutout(bg_added_model, config=config_values)
#
#     # bg_info will be the mean, median, and std, in that order. (see tbridge.mask_cutouts)
#     bg_info = [mask_data["BG_MEAN"], mask_data["BG_MEDIAN"], mask_data["BG_STD"]]
#     noisy_model = tbridge.add_to_noise(convolved_model)
#
#     # print(type(convolved_model), type(noisy_model), type(bg_added_model), type(convolved_model[0]))
#
#     return [convolved_model, noisy_model, masked_model[0]], bg_info, bg_added_model
#
#
# def _reformat_profile_list(profile_list):
#     """ Put the profiles (in a row-format) into the proper column format for saving. """
#
#     reformatted = [[] for i in range(0, len(profile_list[0]))]
#
#     for row in profile_list:
#         for i in range(0, len(row)):
#             reformatted[i].append(row[i])
#
#     return reformatted


# def random_selection(coverage_table, ra_min, ra_max, dec_min, dec_max, band="i"):
#     """ Selects an image based on a random RA and DEC selection. """
#     for n in range(1000):
#         ra, dec = uniform(ra_min, ra_max), uniform(dec_min, dec_max)
#
#         band_rows = []
#         for row in coverage_table:
#             filename = row["Image Filename"]
#             filename_band = filename.split("/")[len(filename.split("/")) - 1].split("_")[0].split("-")[1].lower()
#             if filename_band == band:
#                 band_rows.append(row)
#
#         for row in band_rows:
#             if row["ra_2"] < ra < row["ra_1"] and row["dec_1"] < dec < row["dec_2"]:
#                 return row["Image Filename"]


# def load_positions(location_table, n=100, ra_key="RA", dec_key="DEC", img_filename_key="img_filename",
#                    check_band=False, band_key="band", band="i"):
#     images, ras, decs = location_table[img_filename_key], location_table[ra_key], location_table[dec_key]
#
#     if check_band:
#         bands, band_mask = location_table[band_key], []
#         for i in range(0, len(bands)):
#             if str(bands[i]) == band:
#                 band_mask.append(True)
#             else:
#                 band_mask.append(False)
#         band_mask = array(band_mask)
#
#         images, ras, decs = images[band_mask], ras[band_mask], decs[band_mask]
#
#     index_array = arange(0, len(images), 1, dtype=int)
#     indices = choice(index_array, n, replace=True)
#
#     images, ras, decs = images[indices], ras[indices], decs[indices]
#
#     return array(images, dtype=str), array(ras), array(decs)


# def generate_output_structure(out_dir):
#     """ Generates the structure of the output filesystem, with outdir being the top-level directory.
#     :param out_dir:
#     :return:
#     """
#
#     bare_profile_outdir = out_dir + "bare_profiles/"
#     bgadded_profile_outdir = out_dir + "bgadded_profiles/"
#     noisy_outdir = out_dir + "noisy_profiles/"
#     psf_outdir = out_dir + "psf_profiles/"
#     localsub_outdir = out_dir + "localsub_profiles/"
#
#     if not os.path.isdir(out_dir):
#         os.mkdir(out_dir)
#     for directory in (bare_profile_outdir, bgadded_profile_outdir, noisy_outdir, psf_outdir, localsub_outdir):
#         if not os.path.isdir(directory):
#             os.mkdir(directory)
#
#     return bare_profile_outdir, bgadded_profile_outdir, noisy_outdir, localsub_outdir, psf_outdir


# def trim_hdulist(input_filename, indices, output_filename="out.fits"):
#     """ Trims an HDUList based on a set of user-provided indices
#     Args:
#         input_filename: The filename
#     :param indices:
#     :param output_filename:
#     :return: HDUList of size <= len(indices)
#
#     USAGE
#     indices = [1, 4, 5, 8, 9, 10]
#     trim_hdulist("input.fits", indices, output_filename="output.fits")
#     """
#
#     HDUList = fits.open(input_filename)
#     out_hdulist = fits.HDUList()
#     print(len(HDUList))
#     for n in range(0, len(HDUList)):
#         if n in indices:
#             out_hdulist.append(HDUList[n])
#
#     out_hdulist.writeto(output_filename, overwrite=True)


# def extract_profiles_single_row(cutouts, config, bg_info=None):
#     """ Extract profiles for a single row.
#     Args:
#         cutouts: A list of cutouts to extract. (Single row)
#         config: Configuration parameters
#         bg_info: Background info for the bg-added cutout (to maintain proper order in multithreading).
#
#     Returns:
#         arr: list of profiles and background info
#     """
#
#     output_profiles = []
#
#     for i in range(0, len(cutouts)):
#         t = isophote_fitting(cutouts[i], config)
#
#         if len(t) > 0:
#             output_profiles.append(t.to_table())
#
#     if len(output_profiles) == len(cutouts):
#         return output_profiles, bg_info
#     else:
#         return [], None