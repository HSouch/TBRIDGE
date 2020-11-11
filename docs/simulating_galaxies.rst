Simulating Galaxies
====================

`TBriDGE` makes use of both ``Astropy`` methods, as well as some of its own to generate various galaxy models.


To simulate a set of Sersic models based on input parameters, we can call the following function::

    import tbridge
    sersic_models = tbridge.simulate_sersic_models(mags, r50s, ns, ellips,
                                                   config_values, n_models=n)

