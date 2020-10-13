from matplotlib import pyplot as plt
from astropy.table import Table


def single_bin_plot(table_arrays, colours=None):

    return None


def single_prof(x, y, error=None):
    plt.figure(figsize=(8, 6))

    if error is None:
        plt.plot(x, y, color="orange", lw=2)
    else:
        plt.errorbar(x, y, yerr=error)

    plt.tight_layout()
    plt.show()
