"""
This module contains the methods required to run KOE, which is a tool designed to automatically extract profiles
from a directory of images, with a given input catalogue.
"""

import tbridge
import time
import argparse

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Automated extraction of 1-D Profiles of Surface Brightness using " +
                                             "elliptical isophote analysis.",
                                 epilog="Please submit bug reports to https://github.com/HSouch/TBRIDGE")
parser.add_argument("config", type=str, help="Koe configuration file. Use tbridge.dump_default_config_file_koe() if " +
                                             "an unedited config file is required.")

args = parser.parse_args()

# Load in our configuration file.
config = tbridge.load_config_file_koe(args.config)
if config["VERBOSE"]:
    print("KOE - Running with TBRIDGE")


t_init = time.time()
tbridge.koe_pipeline(config)

print("Finished:", str((time.time() - t_init) / 60)[:8], "minutes.")
