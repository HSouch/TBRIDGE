import tbridge
from numpy import arange
from numpy.random import uniform


def test_binning():
    # Generate some random data to test the binning
    col_1, col_2, col_3 = uniform(0, 1, 1000), uniform(9, 20, 1000), uniform(1, 7, 1000)
    step_1, step_2, step_3 = 0.5, 6, 0.5
    bins_1, bins_2, bins_3 = arange(0, 1, step_1), arange(9, 20, step_2), arange(1, 7, step_3)

    print(bins_1, bins_2, bins_3)

    # Put columns into object format and generate the initial bin
    objects = tbridge.generate_objects((col_1, col_2, col_3))
    init_bin = tbridge.Bin(objects=objects, object_column_names=("COL_1", "COL_2", "COL_3"))

    binned_1 = init_bin.rebin(init_bin.index_from_key("COL_1"), bins_1, bin_width=step_1)
    print("Should return 2:", len(bins_1))

    binned_2 = []
    for b in binned_1:
        binned_2.extend(b.rebin(b.index_from_key("COL_2"), bins_2, bin_width=step_2))
    print("Should return 4:", len(binned_2))

    binned_3 = []
    for b in binned_2:
        binned_3.extend(b.rebin(b.index_from_key("COL_3"), bins_3, bin_width=step_3))
    print("Should return 48:", len(binned_3))

    for n in binned_3:
        print(len(n.objects), n.bin_params)


test_binning()
