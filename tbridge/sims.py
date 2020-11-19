import tbridge
import multiprocessing as mp
from multiprocessing import TimeoutError

from numpy import transpose


def pipeline(config_values, max_bins=None, separate_mags=None, provided_bgs=None, progress_bar=False,
             verbose=False, multiprocess_level='obj'):
    """
    Runs the entire simulation pipeline assuming certain data exists.
    :param config_values: Values from properly loaded configuration file.
    :param max_bins: The number of bins to process (useful if running tests).
    :param separate_mags: Optional array of magnitudes.
    :param provided_bgs: A set of provided background cutouts [OPTIONAL].
    :param progress_bar: Have a TQDM progress bar.
    :param verbose: Have command-line output printing out at various steps
    :param multiprocess_level: Where in the simulations to divide into cores
        'obj'  - Divide at the object level, where each core handles a single object in each bin.
        'bin'  - Divide at the bin level, so each core is responsible for a single bin
        'none' - Do not do any multiprocessing at all (SLOW).

        The object level is less memory intensive, but bins are processed one by one instead of simultaneously.

    EXAMPLE USAGE:

    config_values = tbridge.load_config_file("path/to/config/file.tbridge")
    tbridge.pipeline(config_values, max_bins=10)

    """

    binned_objects = tbridge.bin_catalog(config_values["CATALOG"], config_values)
    max_bins = len(binned_objects) if max_bins is None else max_bins

    if multiprocess_level == 'bin':
        pool = mp.Pool(processes=config_values["CORES"])
        results = [pool.apply_async(_process_bin, (b, config_values, separate_mags, provided_bgs,
                                                   progress_bar, False))
                   for b in binned_objects[:max_bins]]
        [res.get() for res in results]
    elif multiprocess_level == 'obj':
        for b in binned_objects[:max_bins]:
            _process_bin(b, config_values, separate_mags=separate_mags,
                         provided_bgs=provided_bgs, progress_bar=progress_bar, multiprocess=True)


def _process_bin(b, config_values, separate_mags=None, provided_bgs=None, progress_bar=False, multiprocess=False):
    """
    Process a single bin of galaxies. (Tuned for pipeline usage but can be used on an individual basis.
    :param b: Bin to obtain full profiles from.
    :param config_values: Values from properly loaded configuration file.
    :param separate_mags: Optional array of magnitudes.
    :param provided_bgs: Array of provided backgrounds
    :param progress_bar: Use a TQDM progress bar (note with multithreading this might get funky).
    :param multiprocess: Run in multiprocessing mode.
        This means both the model creation AND the
    :return:
    """

    verbose = config_values["VERBOSE"]
    use_alarm, alarm_time = config_values["USE_ALARM"], config_values["ALARM_TIME"]

    # Load in information
    keys, columns = b.return_columns()
    mags, r50s, ns, ellips = tbridge.pdf_resample(tbridge.structural_parameters(keys, columns), resample_size=1000)
    if separate_mags is not None:
        mags = tbridge.pdf_resample(separate_mags, resample_size=len(r50s))[0]

    if multiprocess:
        if verbose:
            print("Simulating Models for: ", b.bin_params)

        with mp.Pool(processes=config_values["CORES"]) as pool:
            models = tbridge.simulate_sersic_models(mags, r50s, ns, ellips,
                                                    config_values, n_models=config_values["N_MODELS"])

            results = [pool.apply_async(_simulate_single_model, (models[i], config_values, provided_bgs))
                       for i in range(0, len(models))]
            model_list = [res.get() for res in results]
        bg_info = []
        for i in range(0, len(model_list)):
            # Every row is going to be a tuple with the list of model images, and the background info for that row.
            row = model_list[i]
            bg_info.append(row[1])
            model_list[i] = row[0]

        bg_info = transpose(bg_info)

        # Get all profile lists from our developed models.
        if verbose:
            print("Extracting Profiles for: ", b.bin_params)

        with mp.Pool(processes=config_values["CORES"]) as pool:
            results = [pool.apply_async(tbridge.extract_profiles_single_row,
                                        (model_list[i], config_values)) for i in range(0, len(model_list))]

            full_profile_list = []
            try:
                full_profile_list = [res.get(timeout=int(config_values["ALARM_TIME"] * 5)) for res in results]
            except TimeoutError:
                print("Timeout error reached! Continuing (I think).")

        # If nothing worked just go to the next bin
        if full_profile_list is None or len(full_profile_list) == 0:
            return

        # Trim all empty arrays from the profile list
        profile_list = [x for x in full_profile_list if len(x) > 0]

        # Reformat into a column-format
        profile_list = _reformat_profile_list(profile_list)

    # If not, simulate the bin serially
    else:
        if verbose:
            print("Simulating Models for: ", b.bin_params)
        # Generate all Sersic models
        models = tbridge.simulate_sersic_models(mags, r50s, ns, ellips, config_values,
                                                n_models=config_values["N_MODELS"])
        # Generate BG added models in accordance to whether a user has provided backgrounds or not
        if provided_bgs is None:
            bg_added_models, convolved_models = tbridge.add_to_locations_simple(models[:], config_values)
            bg_added_models, bg_info = tbridge.mask_cutouts(bg_added_models)
        else:
            convolved_models = tbridge.convolve_models(models, config_values)
            bg_added_models = tbridge.add_to_provided_backgrounds(convolved_models, provided_bgs)
            bg_added_models, bg_info = tbridge.mask_cutouts(bg_added_models)

        noisy_models = tbridge.add_to_noise(convolved_models)

        if verbose:
            print("Extracting Profiles for: ", b.bin_params)

        profile_list = tbridge.extract_profiles((convolved_models, noisy_models, bg_added_models), config_values,
                                                progress_bar=progress_bar, use_alarm=use_alarm, alarm_time=alarm_time)
    if verbose:
        print("Profiles extracted, wrapping up: ", b.bin_params)

    # Estimate backgrounds and generate bg-subtracted profile list
    backgrounds = bg_info[1]
    bgsub_profiles = tbridge.subtract_backgrounds(profile_list[2], backgrounds)
    profile_list.append(bgsub_profiles)

    # Only save the profile in this bin if we have at least 1 set of profiles to save
    if len(profile_list[0]) > 0:
        # Save profiles
        tbridge.save_profiles(profile_list, bin_info=b.bin_params,
                              outdir=config_values["OUT_DIR"], keys=["bare", "noisy", "bgadded", "bgsub"])


def _simulate_single_model(sersic_model, config_values, provided_bgs=None):
    # Generate BG added models in accordance to whether a user has provided backgrounds or not
    model = [sersic_model]
    if provided_bgs is None:
        bg_added_model, convolved_model = tbridge.add_to_locations_simple(model, config_values)
        bg_added_model, bg_info = tbridge.mask_cutouts(bg_added_model)
    else:
        convolved_model = tbridge.convolve_models(model, config_values)
        bg_added_model = tbridge.add_to_provided_backgrounds(convolved_model, provided_bgs)
        bg_added_model, bg_info = tbridge.mask_cutouts(bg_added_model)

    # bg_info will be the mean, median, and std, in that order. (see tbridge.mask_cutouts)
    bg_info = [bg_info[0][0], bg_info[1][0], bg_info[2][0]]
    noisy_model = tbridge.add_to_noise(convolved_model)

    # print(type(convolved_model), type(noisy_model), type(bg_added_model), type(convolved_model[0]))

    return [convolved_model[0], noisy_model[0], bg_added_model[0]], bg_info


def _reformat_profile_list(profile_list, required_length=3):

    reformatted = [[] for i in range(0, len(profile_list[0]))]

    for row in profile_list:
        for i in range(0, len(row)):
            reformatted[i].append(row[i])

    return reformatted
