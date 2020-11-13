============================
Making a Simulation Pipeline
============================

When simulating bins of galaxies to test profile extractions, there is a four step process to
follow:

- Binning your object catalogue
- Simulating galaxies
- Adding these galaxies to various backgrounds
- Extracting and saving profiles

We start as usual by loading in a config file::

    import tbridge
    config_params = tbridge.load_config_file("path/to/config.tbridge")
    # or
    config_params = tbridge.default_config_params()

If you are binning by mass, redshift, and star-formation probability, you can use a built in function
to bin your objects::

    binned_objects = tbridge.bin_catalog(config_values["CATALOG"], config_values)

