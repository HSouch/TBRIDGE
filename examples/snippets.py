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
