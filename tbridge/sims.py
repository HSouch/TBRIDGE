import tbridge
import multiprocessing as mp
import warnings


def pipeline(config_values, max_bins=None, separate_mags=None):
    """
    Runs the entire simulation pipeline assuming certain data exists.
    :param config_values: Values from properly loaded configuration file
    :param max_bins: The number of bins to process (useful if running tests)
    :param separate_mags: Optional array of magnitudes.

    EXAMPLE USAGE:

    config_values = tbridge.load_config_file("path/to/config/file.tbridge")
    tbridge.pipeline(config_values, max_bins=10)

    """

    binned_objects = tbridge.bin_catalog(config_values["CATALOG"], config_values)

    pool = mp.Pool(processes=config_values["CORES"])
    results = [pool.apply_async(process_bin, (b, config_values, separate_mags)) for b in binned_objects[:]]
    [res.get() for res in results]


def process_bin(b, config_values, separate_mags):
    # Load in information
    keys, columns = b.return_columns()
    mags, r50s, ns, ellips = tbridge.pdf_resample(tbridge.structural_parameters(keys, columns), resample_size=1000)
    if separate_mags is not None:
        mags = tbridge.pdf_resample(separate_mags, resample_size=len(r50s))[0]

    print("Simulating Bin: ", b.bin_params)

    # Generate all models
    models = tbridge.simulate_sersic_models(mags, r50s, ns, ellips, config_values, n_models=config_values["N_MODELS"])
    convolved_models, bg_added_models = tbridge.add_to_locations_simple(models[:], config_values)
    bg_added_models = tbridge.mask_cutouts(bg_added_models)
    noisy_models = tbridge.add_to_noise(convolved_models)

    # Extract profiles
    model_list = tbridge.extract_profiles((convolved_models, noisy_models, bg_added_models), progress_bar=False,
                                          linear=True)
    warnings.filterwarnings("ignore")

    # Only save the profile in this bin if we have at least 1 set of profiles to save
    if len(model_list[0]) > 0:
        # Save profiles
        tbridge.save_profiles(model_list, bin_info=b.bin_params,
                              outdir=config_values["OUT_DIR"], keys=["bare", "noisy", "bgadded"])

