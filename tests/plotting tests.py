import tbridge
from tbridge import plotting as plotter


def single_bin_plot_test():
    profiles_bare = tbridge.tables_from_file("../bin/example_profiles_bare.fits")
    profiles_bgadded = tbridge.tables_from_file("../bin/example_profiles_bgadded.fits")
    profiles_noisy = tbridge.tables_from_file("../bin/example_profiles_noisy.fits")

    print(len(profiles_bare), len(profiles_bgadded), len(profiles_noisy))

    plotter.single_bin_plot([profiles_bare, profiles_bgadded, profiles_noisy], colours=["red", "blue", "green"],
                            ylim=(32, 20), ind_profile_alpha=0.4, medians=True)
