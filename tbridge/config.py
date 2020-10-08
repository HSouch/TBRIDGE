from numpy import arange


def load_config_file(filename, verbose_test=False):
    """
    Loads in a config file for TBRIDGE to run
    :param filename:
    :param verbose_test:
    :return:
    """

    config_values = {}

    config_lines = open(filename, "r").readlines()
    for line in config_lines:
        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue
        splits = line.split("=")
        config_values[splits[0].strip()] = splits[1].strip()

    for n in config_values:
        value = config_values[n]
        if value.lower() == "true":
            config_values[n] = True
            continue
        elif value.lower() == "false":
            config_values[n] = False
            continue

    config_values["SIZE"] = int(config_values["SIZE"])
    config_values["CORES"] = int(config_values["CORES"])

    for n in ("MASS_BINS", "REDSHIFT_BINS", "SFPROB_BINS"):
        """ Turn all bins in numpy aranges (just to simplify the process). Will also add a x_step parameter"""
        value_string = config_values[n].split(",")
        bin_output = arange(float(value_string[0]), float(value_string[1]), float(value_string[2]))

        config_values[n] = bin_output
        config_values[n.split("_")[0] + "_STEP"] = float(value_string[2])

    if verbose_test:
        for n in config_values:
            print(n, config_values[n], type(config_values[n]))

    return config_values


def dump_default_config_file(directory):
    """
    Dumps a default configuration file with all necessary parameters in the directory
    :param directory:
    :return:
    """
    lines = ["# This is the catalog you retrieve object parameters for.",
             "CATALOG             = cat.fits",
             "IMAGE_DIRECTORY     = images/",
             "",
             "# Keys for masses, redshifts, and star-formation probability. These are currently required.",
             "MASS_KEY            = MASSES",
             "Z_KEY               = REDSHIFTS",
             "SFPROB_KEY          = SFPROBS",
             "",
             "# Keys for structural parameters. Magnitudes, half-light radii, Sersic index, ellipticity",
             "MAG_KEY             = i",
             "R50_KEY             = R50S",
             "N_KEY               = SERSIC_NS",
             "ELLIP_KEY           = ELLIPS",
             "",
             "# Cutout size, band, cores to run on",
             "SIZE                = 100",
             "BAND                = i",
             "CORES               = 4",
             "",
             "# Bins to run through. (LOWER BOUND, UPPER BOUND, BIN WIDTH)",
             "# Note that the bins are defined by the LOWER BOUND to LOWER BOUND + BIN WIDTH",
             "MASS_BINS           = 10., 12., 0.4",
             "REDSHIFT_BINS       = 0.1, 0.9, 0.2",
             "SFPROB_BINS         = 0.0, 1, 0.5",
             ]

    with open(directory + "config.tbridge", "w+") as f:
        for n in lines:
            f.write(n + "\n")


# load_config_file("../config.tbridge", verbose_test=False)
# dump_default_config_file("../")
