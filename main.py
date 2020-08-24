import os
import csv
import sys
import math
import json
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

import utils
from pathProc import PathProc


def main():
    # hp = hyperparameters
    # ppm = path processing module
    hp = {"directories": ["/Users/sunghyunlim/PycharmProjects/shq_tda/data/internal", "/home/lims/shq_tda/data"],
          "filenames": ["users_uk_1930_f", "users_uk_1930_m", "users_uk_3140_f", "users_uk_3140_m",
           "users_uk_4150_f", "users_uk_4150_m", "users_uk_5160_f", "users_uk_5160_m",
           "users_uk_6170_f", "users_uk_6170_m", "users_uk_71plus_f", "users_uk_71plus_m"],
           "apoe_gene_dirs": ["36c4d8", "fcfa8e", "ze4e4"],
           "read_raw_path": True,
           "clf_algo": "isof", # or ocsvm
           "methods": ["curv","ss"],
           "levels": [6, 8, 11],
           "max_sigdim": 4,
           "tr_sample_size": -1,
           "gaussian_radius": 5,
           "gaussian_step_size": 0.1,
           "num_clus": 2,
           "clus_subsample_count": 3000,
           "spline_res": 3,
           "filter": None,
           "save_plot": False,
           "show_plot": True}
    ppm = PathProc(hp)
    ppm.fit()


if __name__ == "__main__":
    main()