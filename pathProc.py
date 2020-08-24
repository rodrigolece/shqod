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
from esig import tosig as pathsig
import matplotlib.pyplot as plt
from scipy import signal as spsig
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
# from sklearn.cluster import SpectralClustering
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest


import utils

# path processing module

class PathProc():
    def __init__(self, hp):
        self.hp = hp
        for varname in hp.keys():
            setattr(self, varname, hp[varname])


    def get_data(self):
        """
        Read map, trajectory and number of people per demographic group.
        """

        # Setup
        for i, directory in enumerate(self.directories):
            if os.path.isdir(directory):
                os.chdir(directory)
                break
            if i == len(self.directories) - 1:
                raise RuntimeError("Directory not found")
        csv.field_size_limit(sys.maxsize)

        # Read maps
        try:
            self.map = np.load("map.npy" , allow_pickle=True)
            print("Map reading done.")
        except FileNotFoundError:
            os.chdir("../external")
            self.map = []
            for level in self.levels:
                with open("level" + "0" * (level < 10) + str(level) + "_grid.json") as jsonfile:
                    map_datum = json.load(jsonfile)
                    map = np.array(map_datum["fixed"]["grid_data"]).reshape(map_datum["fixed"]["grid_length"], map_datum["fixed"]["grid_width"])
                    self.map.append(map.T)
            os.chdir("../internal")
            np.save("map.npy", self.map)

        # Read Training Trajectories
        if self.read_raw_path:
            sub_dir = []
            for level in self.levels:
                sub_dir.append("level" + "0" * (level < 10) + str(level) + "_first")
            try:
                self.traj = np.load("traj.npy" , allow_pickle=True)
                self.num_ppl_per_dem = np.load("num_ppl_per_dem.npy", allow_pickle=True)
                print("Trajectory reading done.")
            except FileNotFoundError:
                os.chdir("../external")
                self.traj = []
                self.num_ppl_per_dem = np.zeros((len(self.levels), len(self.filenames)))
                bad_traj_count = 0
                num_files = len(sub_dir) * len(self.filenames)
                file_read_counter = 0
                for level_ind, dir in enumerate(sub_dir):
                    print(f"Reading Directory {dir}")
                    os.chdir(dir)
                    traj_per_dir = []
                    for dem_ind, filename in enumerate(self.filenames):
                        print(f"Reading File {filename}")
                        with open(filename+".csv") as csv_file:
                            traj_per_dem = []
                            for item in csv.DictReader(csv_file):
                                self.num_ppl_per_dem[level_ind, dem_ind] += 1
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

                            file_read_counter += 1
                            utils.print_progress_bar(file_read_counter, num_files, "Reading Trajectory Files... ")
                    self.traj.append(traj_per_dir)
                    os.chdir("..")
                os.chdir("../internal")
                np.save("traj.npy", self.traj)
                np.save("num_ppl_per_dem.npy", self.num_ppl_per_dem)
            self.num_ppl = np.sum(self.num_ppl_per_dem)

            # Read APOE Trajectories
            try:
                self.traj_apoe = np.load("traj_apoe.npy", allow_pickle=True)
                print("APOE Trajectory reading done.")
            except FileNotFoundError:
                os.chdir("../external/apoe")
                bad_traj_count = 0
                self.traj_apoe = [] # shape [[[2darray]]], index by [gene][level][player][time, xy]
                # note that in the below, we access by gene->player->level->traj, but twist the data saving
                # into [gene][level][player][time, xy]
                for gene_str in self.apoe_gene_dirs:
                    traj_apoe_per_gene = [[] for level in self.levels]
                    dirs = os.listdir(gene_str)
                    for dir in dirs: # this dir access is by player, not level. but this is as intended.
                        if dir == ".DS_Store":
                            continue
                        filenames = os.listdir(gene_str + "/"+ dir)
                        for filename in filenames:
                            assert len(filenames) == 8
                            # for level_ind, level in enumerate(self.levels):
                            lvl = int(filename[5:8])
                            try:
                                lvl_ind = self.levels.index(lvl)
                            except ValueError:
                                continue
                            with open(gene_str + "/" + dir + "/" + filename) as jsonfile:
                                pre_traj = json.load(jsonfile)["player"]
                                if type(pre_traj) == type([]):
                                    traj_per_person = np.zeros((len(pre_traj), 2))
                                    for i, traj_point in enumerate(pre_traj):
                                        traj_per_person[i] = np.array((traj_point["x"], traj_point["y"]))
                                else:
                                    print("Bad Trajectory Information")
                                    bad_traj_count += 1
                            traj_apoe_per_gene[lvl_ind].append(traj_per_person)
                    self.traj_apoe.append(traj_apoe_per_gene)
                os.chdir("../../internal")
                self.traj_apoe = np.array(self.traj_apoe)
                np.save("traj_apoe", self.traj_apoe)
        else:
            self.num_ppl_per_dem = np.load("num_ppl_per_dem.npy", allow_pickle=True)
            self.num_ppl = np.sum(self.num_ppl_per_dem)

        # Read AD patient data
        try:
            self.traj_ad = np.load("traj_ad.npy", allow_pickle=True)
            print("AD Trajectory reading done.")
        except FileNotFoundError:
            print("Reading AD Trajectory from Raw Data...")
            os.chdir("../external/patients")
            dirs = os.listdir()
            self.traj_ad = [] # shape [[2darray]], index by [person][level][time, xy]
            for dir in dirs:
                traj_ad_per_dir = [[] for lvl in self.levels]
                if dir[0] == ".":
                    continue
                filenames = os.listdir(dir)
                for filename in filenames:
                    if filename[-4:] != "json":
                        continue
                    try:
                        lvl = int(filename[5:8])
                    except ValueError:
                        continue
                    try:
                        lvl_ind = self.levels.index(lvl)
                    except ValueError:
                        continue
                    with open(f"{dir}/{filename}") as jsonfile:
                        pre_traj = json.load(jsonfile)["player"]
                        if type(pre_traj) == type([]):
                            if len(pre_traj) == 0:
                                print("Uh-oh")
                            traj = np.zeros((len(pre_traj), 2))
                            for i, traj_point in enumerate(pre_traj):
                                traj[i] = np.array([traj_point["x"], traj_point["y"]])
                            traj_ad_per_dir[lvl_ind] = traj
                        else:
                            print("Bad Trajectory Information")
                            bad_traj_count += 1
                self.traj_ad.append(traj_ad_per_dir)
            os.chdir("../../internal")
            np.save("traj_ad", self.traj_ad)
            print("Done.")


    def path_smooth(self, path):
        if isinstance(path, np.ndarray):
            return np.vstack((gaussian_filter(utils.linear_extend(path[:, 0], self.spline_res), self.spline_res*1.67),
                              gaussian_filter(utils.linear_extend(path[:, 1], self.spline_res), self.spline_res*1.67))).T
        elif isinstance(path, list):
            if len(path) == 0:
                return []


    def get_traj_sm(self):
        """
        Smooth trajectories
        """
        try:
            self.traj_sm = np.load("traj_sm.npy" , allow_pickle=True)
            print("Smoothed Trajectory reading done.")
        except FileNotFoundError:
            print("Smoothing Normative Paths...")
            # shape [[[2darr]]], index by [level][demographic][player][time,xy]
            self.traj_sm = [[[self.path_smooth(traj) for traj in traj_per_dem]
                             for traj_per_dem in traj_per_lvl]
                            for traj_per_lvl in self.traj]
            np.save("traj_sm", self.traj_sm)
            print("Done")
        try:
            self.traj_apoe_sm = np.load("traj_apoe_sm.npy" , allow_pickle=True)
            print("Smoothed APOE Trajectory reading done.")
        except FileNotFoundError:
            print("Smoothing APOE Paths...")
            # shape [[[2darr]]], index by [gene][level][player][time,xy]
            self.traj_apoe_sm = [[[self.path_smooth(traj) for traj in traj_per_lvl]
                                 for traj_per_lvl in traj_per_gene]
                                 for traj_per_gene in self.traj_apoe]
            np.save("traj_apoe_sm", self.traj_apoe_sm)

            print("Done")
        try:
            self.traj_ad_sm = np.load("traj_ad_sm.npy" , allow_pickle=True)
            print("Smoothed AD Trajectory reading done.")
        except FileNotFoundError:
            print("Smoothing AD Paths...")
            # shape [[2darr]], index by [player][level][time,xy]
            # Indexing access is different from APOE. I would want to fix this at some point.
            self.traj_ad_sm = [[self.path_smooth(traj) for traj in traj_per_plr]
                               for traj_per_plr in self.traj_ad]
            np.save("traj_ad_sm", self.traj_ad_sm)
            print("Done")


    def get_sig(self, path, map):
        if isinstance(path, np.ndarray):
            return pathsig.stream2logsig(path, self.max_sigdim)
        elif path == []:
            return []

    def get_ss(self, path, map):
        if isinstance(path, np.ndarray):
            return np.array([utils.path_curvature(path), utils.path_length(path), utils.path_dtb(path, map)])
        elif path == []:
            return []

    def get_len(self, path, map):
        if isinstance(path, np.ndarray):
            return np.array([utils.path_length(path)])
        elif path == []:
            return []

    def get_curv(self, path, map):
        if isinstance(path, np.ndarray):
            return np.array([utils.path_curvature(path)])
        elif path == []:
            return []

    def get_dtb(self, path, map):
        if isinstance(path, np.ndarray):
            return np.array([utils.path_dtb(path, map)])
        elif path == []:
            return []


    def get_all_features(self):
        for method in self.methods:
            if not (method in {"len","curv","dtb"}):
                try:
                    setattr(self, method, np.load(f"{method}.npy", allow_pickle=True))
                except FileNotFoundError:
                    # Shape [[2darray]], index by [level][demographic][player, feature dimension]
                    counter = 0
                    feat = []
                    for lvl_ind, traj_per_lvl in enumerate(self.traj_sm):
                        fpl = []
                        for traj_per_dem in traj_per_lvl:
                            fpd = []
                            for traj in traj_per_dem:
                                fe = getattr(self, f"get_{method}")(traj, self.map[lvl_ind])
                                fpd.append(fe)
                                counter += 1
                                utils.print_progress_bar(counter, self.num_ppl, f"Calculating {method}...")
                            fpl.append(np.array(fpd))
                        feat.append(fpl)
                    setattr(self, method, feat)
                    print("Done")
                    np.save(f"{method}.npy", feat)

                try:
                    setattr(self, f"{method}_apoe", np.load(f"{method}_apoe.npy", allow_pickle=True))
                except FileNotFoundError:
                    print(f"Calculating {method} for APOE...")
                    # Shape [[3darray]], index by [gene][player, level, feature dimension]
                    feat = [np.array([[getattr(self, f"get_{method}")(traj, self.map[lvl_ind])
                                       for traj in traj_per_lvl]
                                      for lvl_ind, traj_per_lvl in enumerate(traj_per_gene)])
                                .transpose(1,0,2)
                            for traj_per_gene in self.traj_apoe_sm]
                    setattr(self, f"{method}_apoe", feat)
                    print("Done")
                    np.save(f"{method}_apoe.npy", feat)

                try:
                    setattr(self, f"{method}_ad", np.load(f"{method}_ad.npy", allow_pickle=True))
                except FileNotFoundError:
                    print(f"Calculating {method} for ad...")
                    # shape [[[number]]], index by [player][level][feature dimension]
                    # we don't use array here because gameplay data don't exist for some players
                    feat = [[getattr(self, f"get_{method}")(traj, self.map[lvl_ind])
                                      for lvl_ind, traj in enumerate(traj_per_plr)]
                                     for traj_per_plr in self.traj_ad_sm]
                    setattr(self, f"{method}_ad", feat)
                    print("Done")
                    np.save(f"{method}_ad.npy", feat)


    def train_clf(self, algo, X_tr):
        if algo == "isof":
            clf = IsolationForest(bootstrap = True)
            clf.fit(X_tr)
            prop_outliers = np.sum(clf.predict(X_tr) == -1) / X_tr.shape[0]
            print(f"The proportion of outlier is {prop_outliers}")
            return clf
        elif algo == "ocsvm":
            clf = OneClassSVM()
            clf.fit(X_tr)
            prop_outliers = np.sum(clf.predict(X_tr) == -1) / X_tr.shape[0]
            print(f"The proportion of outlier is {prop_outliers}")
            return clf


    def fit(self):
        # Read and Process data
        self.get_data()
        self.get_traj_sm() # smoothen trajectories
        self.get_all_features() # calculate all features and store in self

        for method in self.hp["methods"]:
            print(f"Training using {method}...")
            setattr(self, f"clf_{method}", [])
            setattr(self, f"clf_tr_err_{method}", [])

            for lvl_ind, lvl in enumerate(self.levels):
                print(f"Training clf at Level {lvl} using {method}...")
                # +1 is not outlier, -1 is outlier
                if method == "len":
                    X_tr_raw = np.vstack(self.ss[lvl_ind])[:,1].reshape(-1, 1)
                elif method == "curv":
                    X_tr_raw = np.vstack(self.ss[lvl_ind])[:,0].reshape(-1, 1)
                elif method == "dtb":
                    X_tr_raw = np.vstack(self.ss[lvl_ind])[:,2].reshape(-1, 1)
                else:
                    X_tr_raw = np.vstack(getattr(self, method)[lvl_ind])
                (N_tr_raw, featdim_tr) = X_tr_raw.shape
                if self.tr_sample_size != -1:
                    print(f"We use {self.tr_sample_size} out of {N_tr_raw} samples for training clf.")
                    tr_ind = np.random.choice(N_tr_raw, self.tr_sample_size, replace=False)
                    X_tr = X_tr_raw[tr_ind]
                else:
                    print(f"Using all samples ({N_tr_raw}) for training clf.")
                    X_tr = X_tr_raw
                clf_now = self.train_clf(self.clf_algo, X_tr)
                X_tr_pred = clf_now.predict(X_tr)
                err_rate = np.sum((X_tr_pred == -1) / X_tr.shape[0])
                print("Done")
                getattr(self, f"clf_{method}").append(clf_now)
                getattr(self, f"clf_tr_err_{method}").append(err_rate)

            # APOE Results
            # Shape [2darray], index by [gene][level, player]
            apoe_result = [np.array([[getattr(self, f"clf_{method}")[lvl_ind]
                           .predict(getattr(self, f"get_{method}")(traj, self.map[lvl_ind]).reshape(1, -1))[0]
                           for traj in traj_per_lvl]
                           for lvl_ind, traj_per_lvl in enumerate(traj_per_gene)])
                           for traj_per_gene in self.traj_apoe_sm]
            setattr(self, f"apoe_result_{method}", apoe_result)
            # Rate of deviation per level
            # Shape [1darray], index by [gene][level]
            apoe_outrates = [np.sum(result_per_gene==-1, axis=1)/result_per_gene.shape[1]
                             for result_per_gene in apoe_result]
            setattr(self, f"apoe_outrates_{method}", apoe_outrates)
            apoe_total_outrate = [np.sum(result_per_gene==-1)/result_per_gene.size
                                  for result_per_gene in apoe_result]
            setattr(self, f"apoe_total_outrate_{method}", apoe_total_outrate)
            print(f"APOE: Rate of outliers for {method} is {apoe_total_outrate}")
            np.save(f"apoe_outrates_{method}", apoe_outrates)
            np.save(f"apoe_total_outrate_{method}", apoe_total_outrate)

            os.chdir("clf_apoe_pics")
            for gene_ind, traj_per_gene in enumerate(self.traj_apoe_sm):
                for lvl_ind, traj_per_lvl in enumerate(traj_per_gene):
                    for pers_ind, traj in enumerate(traj_per_lvl):
                        figtitle = f"(Gene{gene_ind},Lvl{self.levels[lvl_ind]},Person{pers_ind})"
                        if apoe_result[gene_ind][lvl_ind][pers_ind] == -1:
                            figtitle = method + "Outlier" + figtitle
                            color = "red"
                        else:
                            figtitle = method + "Good" + figtitle
                            color = "blue"
                        plt.plot(traj[:, 0], traj[:, 1], color=color)
                        map = np.vstack(np.nonzero(self.map[lvl_ind]))
                        plt.scatter(map[0], map[1], color="gray")
                        plt.savefig(figtitle)
                        plt.clf()
            os.chdir("..")

            # AD Results
            # Shape 2darray, index by [player, level]
            # Non-outlier is +1, outlier is -1.
            # Placeholder prediction for nonexistent play data is denoted by -2.
            # For the -2 thing, we have a for loop expanded like that instead of [f(a) for a in b], etc.
            ad_result = []
            for traj_per_plr in self.traj_ad_sm:
                ad_result_per_plr = []
                for lvl_ind, traj in enumerate(traj_per_plr):
                    if isinstance(traj, np.ndarray):
                        pred = getattr(self, f"clf_{method}")[lvl_ind].predict(
                            getattr(self, f"get_{method}")(traj, self.map[lvl_ind])
                                .reshape(1, -1))[0]
                    elif traj == []:
                        pred = -2
                    ad_result_per_plr.append(pred)
                ad_result.append(ad_result_per_plr)
            ad_result = np.array(ad_result)
            setattr(self, f"ad_result_{method}", ad_result)
            # Rate of deviation per player
            valid_data = np.logical_or(ad_result == 1, ad_result == -1)
            ad_outrates = np.sum(ad_result == -1, axis=0) / np.sum(valid_data, axis=0)
            setattr(self, f"ad_outrates_{method}", ad_outrates)
            ad_total_outrate = np.sum(ad_result == -1) / np.sum(valid_data)
            setattr(self, f"ad_total_outrate_{method}", ad_total_outrate)
            print(f"AD: Rate of outliers for {method} is {ad_total_outrate}")
            np.save(f"ad_outrates_{method}", ad_outrates)
            np.save(f"ad_total_outrate_{method}", ad_total_outrate)
