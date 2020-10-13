# for b in binned_objects[:1]:
#     keys, columns = b.return_columns()
#     zest_mags, r50s, ns, ellips = tbridge.pdf_resample(tbridge.structural_parameters(keys, columns), resample_size=1000)
#     mags = tbridge.pdf_resample(clauds_mags, resample_size=len(r50s))[0]
#
#     print("Generating Models")
#     models = tbridge.simulate_sersic_models(mags, r50s, ns, ellips, config_values, n_models=50)
#
#     print("Adding Models to Backgrounds")
#     convolved_models, bg_added_models = tbridge.add_to_locations_simple(models[:], config_values)
#     bg_added_models = tbridge.mask_cutouts(bg_added_models)
#     noisy_models = tbridge.add_to_noise(convolved_models)
#
#     tbridge.save_cutouts(models, "models.fits")
#     tbridge.save_cutouts(bg_added_models, "bg_added_models.fits")
#     tbridge.save_cutouts(noisy_models, "noisy_models.fits")
#
#     print("Extracting Profiles")
#     model_list = tbridge.extract_profiles((models, noisy_models, bg_added_models), progress_bar=True, linear=False)
#
#     tbridge.save_profiles(model_list, bin_info=b.bin_params,
#                           outdir=config_values["OUT_DIR"], keys=["bare", "noisy", "bgadded"])


# profiles = tbridge.tables_from_file("bin/example_profs.fits")
# interps = tbridge.as_interpolations(profiles)
# median_data = tbridge.get_median(interps, bin_max=tbridge.bin_max(profiles))
# b = tbridge.bootstrap_uncertainty(interps, tbridge.bin_max(profiles), iterations=101)
# tbridge.save_medians(median_data, b)