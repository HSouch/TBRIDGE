import tbridge
import multiprocessing as mp


def pipeline(config_values, max_bins=None, separate_mags=None, linear=True, provided_bgs=None, progress_bar=False,
             verbose=False, multiprocess_level='obj'):
    """
    Runs the entire simulation pipeline assuming certain data exists.
    :param config_values: Values from properly loaded configuration file.
    :param max_bins: The number of bins to process (useful if running tests).
    :param separate_mags: Optional array of magnitudes.
    :param linear:
    :param provided_bgs: A set of provided background cutouts [OPTIONAL].
    :param progress_bar: Have a TQDM progress bar.
    :param verbose: Have command-line output printing out at various steps
    :param multiprocess_level: Where in the simulations to divide into cores
        'obj' - Divide at the object level, where each core handles a single object in each bin.
        'bin' - Divide at the bin level, so each core is responsible for a single bin

        The object level is less memory intensive, but bins are processed one by one instead of simultaneously.

    EXAMPLE USAGE:

    config_values = tbridge.load_config_file("path/to/config/file.tbridge")
    tbridge.pipeline(config_values, max_bins=10)

    """

    binned_objects = tbridge.bin_catalog(config_values["CATALOG"], config_values)
    max_bins = len(binned_objects) if max_bins is None else max_bins

    if multiprocess_level == 'bin':
        pool = mp.Pool(processes=config_values["CORES"])
        results = [pool.apply_async(_process_bin, (b, config_values, separate_mags, linear, provided_bgs, progress_bar,
                                                   verbose, False))
                   for b in binned_objects[:max_bins]]
        [res.get() for res in results]
    elif multiprocess_level == 'obj':
        for b in binned_objects:
            _process_bin(b, config_values, separate_mags=separate_mags, provided_bgs=provided_bgs,
                         progress_bar=progress_bar, verbose=verbose, multiprocess=True)


def _process_bin(b, config_values, separate_mags=None, provided_bgs=None, progress_bar=False,
                verbose=False, multiprocess=False):
    """
    Process a single bin of galaxies. (Tuned for pipeline usage but can be used on an individual basis.
    :param b: Bin to obtain full profiles from.
    :param config_values: Values from properly loaded configuration file.
    :param separate_mags: Optional array of magnitudes.
    :param linear:
    :param provided_bgs:
    :param progress_bar:
    :return:
    """

    # Load in information
    keys, columns = b.return_columns()
    mags, r50s, ns, ellips = tbridge.pdf_resample(tbridge.structural_parameters(keys, columns), resample_size=1000)
    if separate_mags is not None:
        mags = tbridge.pdf_resample(separate_mags, resample_size=len(r50s))[0]

    if verbose:
        print("Simulating Bin: ", b.bin_params)

    # Generate all models
    models = tbridge.simulate_sersic_models(mags, r50s, ns, ellips, config_values, n_models=config_values["N_MODELS"])

    # Generate BG added models in accordance to whether a user has provided backgrounds or not
    if provided_bgs is None:
        convolved_models, bg_added_models = tbridge.add_to_locations_simple(models[:], config_values)
        bg_added_models = tbridge.mask_cutouts(bg_added_models)
    else:
        convolved_models = tbridge.convolve_models(models, config_values)
        bg_added_models = tbridge.add_to_provided_backgrounds(convolved_models, provided_bgs)
        bg_added_models = tbridge.mask_cutouts(bg_added_models)

    noisy_models = tbridge.add_to_noise(convolved_models)

    # Extract profiles
    model_list = tbridge.extract_profiles((convolved_models, noisy_models, bg_added_models), config_values,
                                          progress_bar=progress_bar)

    # Estimate backgrounds and generate bg-subtracted profile list
    backgrounds = tbridge.estimate_background_set(bg_added_models)[1]
    bgsub_profiles = tbridge.subtract_backgrounds(model_list[2], backgrounds)
    model_list.append(bgsub_profiles)

    # Only save the profile in this bin if we have at least 1 set of profiles to save
    if len(model_list[0]) > 0:
        # Save profiles
        tbridge.save_profiles(model_list, bin_info=b.bin_params,
                              outdir=config_values["OUT_DIR"], keys=["bare", "noisy", "bgadded", "bgsub"])
