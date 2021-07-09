import os
from tqdm import tqdm

from shqod import *


project_dir = os.environ["dementia"]  # set in the shell
data_dir = os.path.join(project_dir, "data")
grid_dir = os.path.join(data_dir, "grids")

normative_dir = os.path.join(data_dir, "normative", "all_levels")
apoe_dir = os.path.join(data_dir, "apoe_dataframes")
ad_dir = os.path.join(data_dir, "ad_dataframe")


### ------------------------------------- ###
### Load data --------------------------- ###
### ------------------------------------- ###

# Loader for the normative data (untidy data, see docstring)
norm_loader = UntidyLoader(normative_dir, fmt="csv")

level = 6
gender = "f"
age = 50

print("\nUsing UntidyLoader")
print("Loading file, this is slow")
df = norm_loader.get(level, gender, age)
# this function call is slow because the whole file for level 6 and 'f' is loaded

print("Loading different age, this is fast")
age = 51  # different age
df = norm_loader.get(level, gender, age)
# this is fast because the only operation is filtering (the file is loaded)

# Try loading an invalid level
print("\nTrying to load an invalid level")

level = 7
try:
    norm_loader.get(level, gender, age)
except FileNotFoundError:
    # an exception is thrown
    print("File not found")

# Loader for the smaller groups (tidy data), the apoe genetic
# groups and the AD patients
groups_loader = TidyLoader(apoe_dir, ad_dir)
# We convert the json trajectories to numpy arrays
_ = [groups_loader.json_to_array(df) for df in groups_loader.loaded.values()]

print("\nUsing TidyLoader")
print("Available groups: ", groups_loader.loaded.keys())

df = groups_loader.loaded["ad"]  # select the dementia patients for example

print("\nShape of AD DataFrame: ", df.shape)

# Select an individual level instead, still for the dementia patients
level = 6
df = groups_loader.get(level)["ad"]
print(f"Shape of AD DataFrame for level {level}: ", df.shape)


### ------------------------------------- ###
### Calculate feaures ------------------- ###
### ------------------------------------- ###

# We convert the json trajectories to numpy arrays
_ = [norm_loader.json_to_array(df) for df in norm_loader.loaded.values()]

level = 6
gender = "f"

hp = dict()
hp["grid_dir"] = grid_dir
hp["spline_res"] = 3
hp["max_sigdim"] = 4

print("\nUsing TrajProcessor")
proc = TrajProcessor(level, gender, **hp)

# Calculate Finn's features for df (ad_patients) all in one go
print("Calculating features for AD patients")

smooth_feat_types = ["len", "curv", "dtb", "sig"]
feat_df = proc.get_smooth_features(df, smooth_feat_types)

print("\nSome results")
print(feat_df.iloc[:, :9].describe())  # print description for a subset of cols


# Calculate the features that use the normative data
print("\nUsing NormativeProcessor")
norm = NormativeProcessor(norm_loader, level, gender, **hp)
# note, the processor take the normative trajectories loader

print("\nCalculating features for AD patients")
df = groups_loader.get(level, gender)["ad"]

# Initially the features cannot be used because the reference od matrix has
# not been set
print("\nTrying to calcucate the features without reference")
try:
    norm.get_mob(df.iloc[0].trajectory_data)
except:
    print("Exception ocurred, normative OD matrix has not been set")

# We load the od matrix from the normative population of the selected age
print("\nSet the data for a given age")
age = 50
mat, N = norm.normative_od_matrix_for_age(age)

# set the data
norm.normative_mat = (mat, N)

normative_feat_types = ["fro", "inf", "sum_match", "mob"]
feat_df = norm.get_coarse_features(df, normative_feat_types)

print("\nSome results")
print(feat_df.describe())  # print description for a subset of cols
