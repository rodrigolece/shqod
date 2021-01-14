import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from shqod import *
from shqod.smoothing import smooth


project_dir = os.environ['dementia']  # set in the shell
data_dir = os.path.join(project_dir, 'data')
grid_dir = os.path.join(data_dir, 'grids')

normative_dir = os.path.join(data_dir, 'normative', 'all_levels')
apoe_dir = os.path.join(data_dir, 'apoe_dataframes')
ad_dir = os.path.join(data_dir, 'ad_dataframe')


### ------------------------------------- ###
### Load data --------------------------- ###
### ------------------------------------- ###

# Loader for the normative data (untidy data, see docstring)
norm_loader = UntidyLoader(normative_dir)

level = 6
gender = 'f'
age = 50

print('\nUsing UntidyLoader')
print('Loading file, this is slow')
df = norm_loader.get(level, gender, age)
# this function call is slow because the whole file for level 6 and 'f' is loaded

print('Loading different age, this is fast')
age = 51  # different age
df = norm_loader.get(level, gender, age)
# this is fast because the only operation is filtering (the file is loaded)

# Try loading an invalid level
print('\nTrying to load an invalid level')

level = 7
try:
    norm_loader.get(level, gender, age)
except FileNotFoundError:
    # an exception is thrown
    print('File not found')

# Loader for the smaller groups (tidy data), the apoe genetic
# groups and the AD patients
groups_loader = TidyLoader(apoe_dir, ad_dir)

print('\nUsing TidyLoader')
print('Available groups: ', groups_loader.loaded.keys())

df = groups_loader.loaded['ad']  # select the dementia patients for example
print('\nShape of AD DataFrame: ', df.shape)

# Select an individual level instead, still for the dementia patients
level = 6
df = groups_loader.get_level(level)['ad']
print(f'Shape of AD Datframe for level {level}: ', df.shape)


### ------------------------------------- ###
### Calculate feaures ------------------- ###
### ------------------------------------- ###

level = 6
gender = 'f'

hp = dict()
hp['grid_dir'] = grid_dir
hp['spline_res'] = 3
hp['max_sigdim'] = 4

print('\nUsing TrajProcessor')
proc = TrajProcessor(level, gender, **hp)

# Calculate Finn's features for df (ad_patients) all in one go
print('Calculating features for AD patients')

nb_individuals = len(df)
nb_features = 3 + 8  # 8 for signature

# Preallocate the array to store the results
features = np.zeros((nb_individuals, nb_features))
ts = trajecs_from_df(df)  # returns a generator

for row, t in enumerate(ts):
    st = smooth(t)
    features[row] = [proc.get_len(st), proc.get_curv(st),
                     proc.get_dtb(st)] + proc.get_sig(st).tolist()

# We load this into a DataFrame
cols = ['len', 'curv', 'dtb'] + ['sig' + str(i) for i in range(1, 9)]
feat_df = pd.DataFrame(features, columns=cols)

print('\nSome results')
print(feat_df[cols[:5]].describe())  # print description for a subset of cols


# Calculate the features that use the normative data
print('\nUsing NormativeProcessor')
norm = NormativeProcessor(norm_loader, level, gender, **hp)
# note, the processor take the normative trajectories loader

# Initially the features cannot be used because the reference od matrix has
# not been set
print('\nTrying to calcucate the features without reference')
try:
    norm.get_mob(t)
except:
    print('Exception ocurred, normative OD matrix has not been set')

# We load the od matrix from the normative population of the selected age
print('\nSet the data for a given age')
age = 20
mat, N = norm.normative_od_matrix_for_age(age)

# set the data
norm.normative_mat = (mat, N)

print('\nCalculating features for AD patients')
features = np.zeros((nb_individuals, 4))

for row, t in enumerate(trajecs_from_df(df)):
    features[row] = [norm.get_fro(t), norm.get_inf(t),
                     norm.get_sum_match(t), norm.get_mob(t)]

feat_df = pd.DataFrame(features, columns=['Fro', 'Inf', 'Sum', 'Mob'])
print('\nSome results')
print(feat_df.describe())  # print description for a subset of cols
