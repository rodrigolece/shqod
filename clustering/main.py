import os
import csv
import sys
import math
import json
import heapq
import random
import datetime
import numpy as np
import gap_statistic
from esig import tosig
import matplotlib.pyplot as plt
from scipy import signal as spsig
from sklearn.cluster import SpectralClustering

import utils

global directories
directories = ["/Users/sunghyunlim/PycharmProjects/shq_tda/data/internal", "/home/lims/shq_tda/data"]


def main():
    # Setup
    global directories
    for i, directory in enumerate(directories):
        if os.path.isdir(directory):
            os.chdir(directory)
            break
        if i == len(directories) - 1:
            raise RuntimeError("Directory not found")
    csv.field_size_limit(sys.maxsize)
    cluster_k_upper_bound = 10
    max_sigdim = 4
    levels = [6, 8, 11]
    filenames = ["users_uk_1930_f", "users_uk_1930_m", "users_uk_3140_f", "users_uk_3140_m",
                 "users_uk_4150_f", "users_uk_4150_m", "users_uk_5160_f", "users_uk_5160_m",
                 "users_uk_6170_f", "users_uk_6170_m", "users_uk_71plus_f", "users_uk_71plus_m"]
    gaussian_radius = 5
    gaussian_step_size = 0.1
    clus_K = 2
    directories = []
    ppl_count = np.zeros((len(levels), len(filenames))) # number of people per demographic group in each level
    for level in levels:
        directories.append("level" + "0" * (level < 10) + str(level))

    # Read trajectories
    try:
        traj = np.load("traj.npy" , allow_pickle=True)
        ppl_count = np.load("ppl_count.npy", allow_pickle=True)
        if len(traj) != len(levels):
            raise FileNotFoundError
    except FileNotFoundError:
        os.chdir("../external")
        traj = []
        bad_traj_count = 0
        for level_ind, dir in enumerate(directories):
            print(f"Reading Directory {dir}")
            os.chdir(dir)
            traj_per_dir = []
            for dem_ind, filename in enumerate(filenames):
                print(f"Reading File {filename}")
                with open(filename+".csv") as csv_file:
                    traj_per_dem = []
                    for item in csv.DictReader(csv_file):
                        ppl_count[level_ind, dem_ind] += 1
                        pre_traj = json.loads(item["trajectory_data"])["player"]
                        traj_per_person = np.zeros((len(pre_traj), 2))
                        if type(pre_traj) == type([]):
                            for i, traj_point in enumerate(pre_traj):
                                traj_per_person[i] = np.array((traj_point["x"], traj_point["y"]))
                            traj_per_dem.append(traj_per_person)
                        else:
                            print("Bad Trajectory Information")
                            bad_traj_count += 1
                    traj_per_dir.append(traj_per_dem)
            traj.append(traj_per_dir)
            os.chdir("..")
        os.chdir("../internal")
        np.save("traj.npy", traj)
        np.save("ppl_count.npy", ppl_count)


    # Read grids
    try:
        grid = np.load("grid.npy" , allow_pickle=True)
        if len(grid) != len(levels):
            raise FileNotFoundError
    except FileNotFoundError:
        os.chdir("../external")
        grid = []
        for level in levels:
            with open("level" + "0" * (level < 10) + str(level) + "_grid.json") as jsonfile:
                grid_datum = json.load(jsonfile)
                inverted_grid = 1-np.array(grid_datum["fixed"]["grid_data"]).reshape(grid_datum["fixed"]["grid_length"], grid_datum["fixed"]["grid_width"])
                grid.append(inverted_grid)
        os.chdir("../internal")
        np.save("grid.npy", grid)


    # Compute path signatures
    try:
        all_sig = np.load("all_sig.npy" , allow_pickle=True)
        all_label = np.load("all_label.npy", allow_pickle=True)
        if (len(all_sig) != len(levels)) | (len(all_label) != len(levels)):
            raise FileNotFoundError
    except FileNotFoundError:
        os.chdir("../external")
        sigdim = tosig.logsigdim(2, max_sigdim)
        all_sig = []
        all_label = []
        for level_ind, level in enumerate(levels):
            print(f"Signature computation for {level}")
            num_per_dem = int(np.sum(ppl_count[level_ind]))
            sig_per_level = np.zeros((num_per_dem, sigdim))
            label_per_level = np.zeros((num_per_dem, 2))

            grid_curr = grid[level_ind]
            gaussian_filter = utils.gaussian_2d_filter(gaussian_radius, gaussian_step_size)
            convolved_grid = spsig.convolve(grid_curr, gaussian_filter)[gaussian_radius:-gaussian_radius, gaussian_radius:-gaussian_radius]
            convolved_grid += 0.01
            narrow_filter = 1/convolved_grid

            people_counter = 0
            for dem_ind, paths_per_dem in enumerate(traj[level_ind]):
                print(f"Sig for {dem_ind}")
                for person_ind, path in enumerate(paths_per_dem):
                    path = path.astype(int)
                    narrowness_along_path = narrow_filter[path[:,1], path[:,0]]
                    reweighted_velocity = np.matmul(narrowness_along_path[1:].reshape(-1,1),np.ones((1,2))) * utils.path_velocity(path)

                    reweighted_path = utils.path_integrate(reweighted_velocity)
                    sig_per_level[people_counter] = tosig.stream2logsig(reweighted_path, max_sigdim)
                    label_per_level[people_counter] = np.array([dem_ind, person_ind])
                    people_counter += 1

            all_sig.append(sig_per_level)
            all_label.append(label_per_level)
        os.chdir("../internal")
        np.save("all_sig.npy", all_sig)
        np.save("all_label.npy", all_label)


    # Spectral clustering
    all_ppl_by_clus = []
    for level_ind, level in enumerate(levels):
        sig_per_level = all_sig[level_ind]
        subsample_indices = np.random.choice(sig_per_level.shape[0], 3000, replace=False)
        sig_per_level_subsample = sig_per_level[subsample_indices]
        clus_algo = SpectralClustering(n_clusters=clus_K, assign_labels="discretize", random_state=0)
        # clus_algo = SpectralClustering(n_clusters=clus_K, eigen_solver='arpack', affinity="nearest_neighbors")
        clus_algo.fit(sig_per_level_subsample)
        clus_label_arr = clus_algo.labels_
        clus_all_label = list(set(clus_label_arr))
        ppl_by_clus = []
        for label in clus_all_label:
            ppl_by_clus.append([])
        for person_ind, person_clus_label in enumerate(clus_label_arr):
            ppl_by_clus[clus_all_label.index(person_clus_label)].append(all_label[level_ind][subsample_indices[person_ind]])

        max_plotcount = 10
        for ppl_clus_ind, ppl_list in enumerate(ppl_by_clus):
            uuu = min(max_plotcount, len(ppl_list))
            for i in range(uuu):
                mypath = traj[level_ind][int(ppl_list[i][0])][int(ppl_list[i][1])]
                plt.plot(mypath[:,0], mypath[:,1])
            plt.savefig(f"Traj by Clus, Level={level}, ({ppl_clus_ind}) ({uuu} curves).png")
            plt.clf()

        all_ppl_by_clus.append(ppl_by_clus)
    np.save("all_ppl_by_clus", all_ppl_by_clus)

if __name__ == "__main__":
    main()