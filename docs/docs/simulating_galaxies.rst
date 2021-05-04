Simulating Galaxies
====================

``TBriDGE`` makes use of both ``Astropy`` methods, as well as some of its own to generate various galaxy models.


To simulate a set of Sersic models based on input parameters, we can call the following function::

    import tbridge
    sersic_models = tbridge.simulate_sersic_models(mags, r50s, ns, ellips,
                                                   config_values, n_models=n)


Where ``mags, r50s, ns, ellips`` are arrays containing the magnitudes, half-light-radii (in arcseconds),
Sersic indices, and ellipticities. ``config_values`` are the configuration file values (see config.rst).


Now let's test adding a method to this madness: :meth:`~tbridge.data.get_backgrounds`.