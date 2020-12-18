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
