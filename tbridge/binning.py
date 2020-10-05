from astropy.table import Table
from numpy import array, copy, append


class Bin:
    def __init__(self, objects=None, object_column_names=None, bin_params=None):
        if bin_params is None:
            bin_params = []
        self.objects = objects
        self.object_column_names = object_column_names
        self.bin_params = bin_params

    def rebin(self, param_index, low_bounds, bin_width, number_threshold=4):
        """
        Returns a list of Bin objects after re-binning based on a certain value and bin parameters.
        Bins specified by the lower bounds and bin width.
        :param param_index: Which index in the objects contains the desired value.
        :param low_bounds: Lower bounds on the bins.
        :param bin_width: The width of the bins.
        :param number_threshold: The required number of objects a bin needs to have to be saved to the list.
        :return: List of re-binned Bin objects.
        """
        new_bins = []
        top_level_params = self.bin_params
        for low in low_bounds:
            this_bin = []

            for obj in self.objects:
                if low < obj[param_index] < low + bin_width:
                    this_bin.append(obj)
            if len(this_bin) <= number_threshold:
                continue

            new_bin = Bin(objects=this_bin, object_column_names=self.object_column_names,
                          bin_params=append(copy(top_level_params), (low, low + bin_width)))
            new_bins.append(new_bin)

        return new_bins

    def return_columns(self):
        """
        Returns a dictionary of columns (turns the object list back into a column format)
        """
        columns = []
        for i in range(0, len(self.object_column_names)):
            columns.append([])
        for obj in self.objects:
            for i in range(0, len(obj)):
                columns[i].append(obj[i])

        out = {}
        for i in range(0, len(self.object_column_names)):
            out[self.object_column_names[i]] = columns[i]
        return out

    def index_from_key(self, key=""):
        """ Return the index of the column name key specified."""
        return self.object_column_names.index(key)


def generate_objects(params):
    """
    Reorganizes the table columns into an object-format
    :param params: A list of table columns
    :return:
    """
    objects = []
    for i in range(0, len(params[0])):
        this_obj = []
        for j in range(0, len(params)):
            this_obj.append(params[j][i])
        objects.append(this_obj)
    return objects


def bin_catalog(catalog_filename, config_values):
    """
    Bin the inputted catalog.
    :param catalog_filename: Catalog of objects.
    :param config_values: Values from inputted config file or user inputs.
    :return: full_bins - a list of Bin objects for further processing
    """

    # Read HST-ZEST catalog into memory
    catalog = Table.read(catalog_filename, format="fits")

    mags = array(catalog[config_values["MAG_KEY"]])
    r50s = array(catalog[config_values["R50_KEY"]])
    ns = array(catalog[config_values["N_KEY"]])
    ellips = array(catalog[config_values["ELLIP_KEY"]])
    masses = array(catalog[config_values["MASS_KEY"]])
    sfprobs = array(catalog[config_values["SFPROB_KEY"]])
    redshifts = array(catalog[config_values["Z_KEY"]])

    mass_bins, mass_step = config_values["MASS_BINS"], config_values["MASS_STEP"]
    redshift_bins, redshift_step = config_values["REDSHIFT_BINS"], config_values["REDSHIFT_STEP"]
    sfprob_bins, sfprob_step = config_values["SFPROB_BINS"], config_values["SFPROB_STEP"]

    catalog_objects = generate_objects((mags, r50s, ns, ellips, masses, redshifts, sfprobs))
    column_names = ["MAGS", "R50S", "NS", "ELLIPS", "MASSES", "REDSHIFTS", "SFPROBS"]
    init_bin = Bin(objects=catalog_objects, object_column_names=column_names, bin_params=[])

    mass_bins = init_bin.rebin(4, mass_bins, mass_step)
    mz_bins, full_bins = [], []
    for cat_bin in mass_bins:
        mz_bins.extend(cat_bin.rebin(5, redshift_bins, redshift_step))
    for cat_bin in mz_bins:
        full_bins.extend(cat_bin.rebin(6, sfprob_bins, sfprob_step))

    return full_bins
