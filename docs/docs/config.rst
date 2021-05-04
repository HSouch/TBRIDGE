Setting up TBriDGE Config
=========================

``TBriDGE`` is a simulation suite that is designed to be customized and tuned by the user.
Because of this, loading in a set of configuration values is the first step in using the package.

We can load in a set of default configuration values this way::

    import tbridge
    config_params = tbridge.default_config_params()

This will be in the form of a dictionary, so parameters can be accessed and set as expected::

    value = config_params["KEY"]
    config_params["KEY"] = new_value

The best practice however is to have a configuration file with customized parameters, and edit those
manually from session to session. Users can dump a default configuration file using the following::

    tbridge.dump_default_config_file()

And to load this file (or any configuration file), we use this command::

    config_params = tbridge.load_config_file("path/to/config/file.tbridge")

Users can also load config file parameters from a file stored online by inputting a URL. Users can
test this by loading in the following file::

    tbridge.load_config_file("https://raw.githubusercontent.com/HSouch/HelpfulSnippets/master/config.tbridge")

The parameters in the config file are as follows::

    CATALOG             = Catalog of structural parameters
    IMAGE_DIRECTORY     = Input image directory
    PSF_FILENAME        = /home/harrison/Desktop/Research/compiled_psfs/i_psfs.fits
    OUT_DIR             = out_sim/
    SAVE_CUTOUTS        = none
    CUTOUT_FRACTION     = 0.5


