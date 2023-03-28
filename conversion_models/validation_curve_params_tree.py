import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def add_feature(data_before, adding):
    if (len(data_before.shape)<2):
        data_before = data_before.reshape(-1,1)
    if (len(adding.shape)<2):
        adding = adding.reshape(-1,1)
        
    print(f'Updating shape: {data_before.shape}, {adding.shape}')        
    data_after = np.hstack((data_before, adding))
    print(f'shape of data after adding feature: {data_after.shape}')
    if data_after.shape[1]>1:
        return data_after
    else:
        print('Error')
        return 1

# DATASET
#--------
pattern = ['/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr',
          '__xarray_dataarray_variable__']

allsr355 = xr.open_dataset(f'{pattern[0]}355-3000-4000.nc')[pattern[1]].values
allsr532 = xr.open_dataset(f'{pattern[0]}532-3000-4000.nc')[pattern[1]].values

    # PARAMETERS
    #-----------
alt = xr.open_dataset(f'{pattern[0]}355-3000-4000.nc')['range'].values
time = xr.open_dataset(f'{pattern[0]}355-3000-4000.nc')['time'].values
mat_alt = np.tile(alt, (allsr355.shape[0],1))
mat_time = np.tile(time, (allsr355.shape[1], 1)).T

n_features = 3 
preprocess = False

# GENERATE FEATURES
#------------------

if n_features == 2 : 
    data = add_feature(allsr355.ravel(), mat_alt.ravel())
    target = allsr532.ravel()
    # clean data, target from Nan/negative values 
    #--------------------------------------------

    mask = np.logical_and(~np.isnan(data[:, 0]), ~np.isnan(target))
    data = data[mask, :]
    target = target[mask]
else: 
    # add 3e feature of X
    allsr355_after = add_feature(allsr355.ravel(), mat_alt.ravel()) # 1st and 2nd feature
    from tqdm import tqdm
    X3 = np.zeros(mat_alt.shape)
    print(X3.shape)
    X3[0,:] = np.nan
    for j in tqdm(range(1, mat_alt.shape[1])):
        X3[:,j] = allsr355[:,j-1] + allsr355[:,j]*(mat_alt[:,j] - mat_alt[:,j-1])
    print(X3)
    allsr355_after = add_feature(allsr355_after, X3.ravel())
    data = allsr355_after
    target = allsr532.ravel()
    # clean data, target from Nan/negative values 
    #--------------------------------------------

    mask = np.logical_and(~np.isnan(data[:, 0]), ~np.isnan(data[:, 2]), ~np.isnan(target))
    data = data[mask, :]
    target = target[mask]




# split data training & testing
#------------------------------
from sklearn.model_selection import train_test_split, ShuffleSplit

cv = ShuffleSplit(n_splits=5, test_size=0.1)

# create model : Decision Tree Regression
#----------------------------------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

if preprocess : 
    preprocessing = StandardScaler()
    processing = DecisionTreeRegressor()
    model = make_pipeline(preprocessing, processing)
else:
    model = DecisionTreeRegressor()

# # create validation curve to testing performance of parameters
# #-------------------------------------------------------------
# from sklearn.model_selection import validation_curve

# params = {
#     'max_depth' : [3, 7, 15, 30],
#     # 'min_samples_leaf' : (5, 10, 15, 20, 30),
#     'min_samples_split' : [50, 70, 90]
# }

# from sklearn.model_selection import GridSearchCV
# model_grid_search = GridSearchCV(model, param_grid=params, n_jobs=2, cv=cv)
# model_grid_search.fit(data, target)

# data_train, data_test, target_train, target_test = train_test_split(
#     data, target, test_size=0.2, random_state=42
# )
# accuracy = model_grid_search.score(data_test, target_test)
# print(f'Best params: {model_grid_search.best_params_}, \n Accuracy : {accuracy}')

# for key in params.keys():
#     print(key, params[key])
#     if preprocess:
#         train_scores, test_scores = validation_curve(model['decisiontreeregressor'], data, target, cv=cv,
#                 param_name = key, param_range=params[key],
#                 scoring='neg_mean_absolute_error', n_jobs=2)
#         outputname = f'validation_curve_tree_preprocessing_ipral2018_{key}_{n_features}features.png'
#     else:
#         train_scores, test_scores = validation_curve(model, data, target, cv=cv,
#                 param_name = key, param_range=params[key],
#                 scoring='neg_mean_absolute_error', n_jobs=2)
#         outputname = f'validation_curve_tree_ipral2018_{key}_{n_features}features.png'

#     train_errors, test_errors = -train_scores, -test_scores

    # fig, ax = plt.subplots()
    # ax.plot(params[key], train_errors.mean(axis=1), label=f"Training error {key}")
    # ax.plot(params[key], test_errors.mean(axis=1), label=f"Testing error {key}")
    # ax.legend()
    # ax.set(xlabel=f"{key} of decision tree", ylabel="Mean absolute error", 
    #     title="Validation curve for decision tree (ipral2018)")
    # plt.savefig(f'/home/nmpnguyen/conversion_model/{outputname}')
    # # plt.clf()
    # # plt.close()


# create learning curve to testing performance of data training 
#--------------------------------------------------------------
from sklearn.model_selection import learning_curve

train_sizes_range = np.linspace(0.1, 1.0, num=10, endpoint=True)
results = learning_curve(model, data, target, cv=cv,
    train_sizes = train_sizes_range, scoring='neg_mean_absolute_error', n_jobs=2)
train_size, train_scores, test_scores = results[:3]
# Convert the scores into errors
train_errors, test_errors = -train_scores, -test_scores

fig, ax = plt.subplots()
ax.errorbar(train_sizes_range, train_errors.mean(axis=1),
             yerr=train_errors.std(axis=1), label="Training error")
ax.errorbar(train_sizes_range, test_errors.mean(axis=1),
             yerr=test_errors.std(axis=1), label="Testing error")
ax.legend()
# ax2 = ax.twiny()
# ax2.errorbar(train_sizes_range, test_errors.mean(axis=1),
#              yerr=test_errors.std(axis=1), label="Testing error")

plt.xscale("log")
plt.xlabel("Number of samples in the training set")
plt.ylabel("Mean absolute error")
_ = plt.title("Learning curve for decision tree")
plt.savefig(f'/home/nmpnguyen/conversion_model/learning_curve_decision_tree_ipral2018.png')
