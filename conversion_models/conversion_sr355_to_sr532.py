
# coding: utf-8

# # Conversion SR532' à partir de SR355 
# --------------------------------------------------------
# 

# In[1]:


import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path


# In[2]:


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# In[3]:


from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ------------------------------

# In[4]:


def remove_NaN_Inf_values(arrayX, arrayY):
    idsX = np.where(~np.isnan(arrayX)&~np.isinf(arrayX))[0]
    idsY = np.where(~np.isnan(arrayY)&~np.isinf(arrayY))[0]
    mask = np.union1d(idsX, idsY)
    return mask

def get_params_histogram(srlimite, Xdata, Ydata):
    from scipy import stats
    from scipy.optimize import curve_fit
    # objective function for best fit
    def objective(x, a, b):
        return a * x + b
    
#     if (~np.isnan(Xdata)|~np.isinf(Xdata)).sum() > (~np.isnan(Ydata)|~np.isinf(Ydata)).sum():
    mask = remove_NaN_Inf_values(Xdata, Ydata)
    print('A')
    H = np.histogram2d(Xdata[mask], Ydata[mask], bins=100, range = srlimite)
    Hprobas = H[0]*100/len(Ydata[mask])
    noNaNpoints = len(Ydata[mask])
    # create the curve fit
    param, param_cov = curve_fit(objective, Xdata[mask], Ydata[mask])
    print(param, param_cov)

    print(f'nombre de points no-NaN: {noNaNpoints}')
    xedges, yedges = np.meshgrid(H[1], H[2])
#     print(slope, intercept)
#     fitLine = slope * allsr532 + intercept
    return xedges, yedges, Hprobas, param


# In[5]:


def find_nearest_time(time_array, value):
    time_array = pd.to_datetime(time_array)
    idt = (np.abs(time_array - pd.to_datetime(value))).argmin()
    time_value = time_array[idt]
    return idt, time_value


# # Create KFold Cross-validation and Training model

# In[6]:


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

def performance_val(Xinput_val, Yinput_val, Ypred_val):
    residus_val = Yinput_val - Ypred_val
    stats_residus_val = pd.DataFrame(residus_val).describe()#[['mean', 'std']]
    return stats_residus_val #mean_residus, std_residus

def DecisionTree_model(Xinput_train, Yinput_train, Xinput_val, Yinput_val):#
    from sklearn.tree import DecisionTreeRegressor as DTR
    tree = DTR(min_samples_leaf=5)
    tree = tree.fit(Xinput_train, Yinput_train)
    Ypred_val = tree.predict(Xinput_val)
    stats_residus_val = performance_val(Xinput_val, Yinput_val, Ypred_val)
#     print(f'Mean of residus = {mean_residus}')
#     print(f'STD of residus = {std_residus}')
    return Ypred_val, stats_residus_val, tree

def LinearRgression_model(Xinput_train, Yinput_train, Xinput_val, Yinput_val):
    from sklearn.linear_model import LinearRegression as LR
    LRmodel = LR(fit_intercept=True)
    LRmodel = LRmodel.fit(Xinput_train, Yinput_train)
    Ypred_val = LRmodel.predict(Xinput_val)
    stats_residus_val = performance_val(Xinput_val, Yinput_val, Ypred_val)
    return Ypred_val, stats_residus_val, LRmodel


# ### Dataset and Parameters

# In[7]:


# DATASET
#--------
pattern = ['/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr',
          '__xarray_dataarray_variable__']

# allsr355 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allatb355-3000-4000.nc')['__xarray_dataarray_variable__'].values
# allsr532 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allatb532-3000-4000.nc')['__xarray_dataarray_variable__'].values

allsr355 = xr.open_dataset(f'{pattern[0]}355-3000-4000.nc')[pattern[1]].values
allsr532 = xr.open_dataset(f'{pattern[0]}532-3000-4000.nc')[pattern[1]].values

# PARAMETERS
#-----------
alt = xr.open_dataset(f'{pattern[0]}355-3000-4000.nc')['range'].values
time = xr.open_dataset(f'{pattern[0]}355-3000-4000.nc')['time'].values


# In[41]:


alt 


# In[38]:


mat_alt = np.tile(alt, (allsr355.shape[0],1))
mat_time = np.tile(time, (allsr355.shape[1], 1)).T

print(mat_alt.shape, mat_time.shape)


# ------------------

# In[41]:


from sklearn.model_selection import train_test_split, ShuffleSplit

data = add_feature(allsr355.ravel(), mat_alt.ravel())
target = allsr532.ravel()

# clean data, target from Nan/negative values 
#--------------------------------------------

mask = np.logical_and(~np.isnan(data[:, 0]), ~np.isnan(target))
data = data[mask, :]
target = target[mask]


# split data training & testing
#------------------------------
cv = ShuffleSplit(n_splits=5, test_size=0.35)


# In[61]:


(data[:,0] < 0).sum()/len(data[:,0]) *100, (target<0).sum()/len(target)


# In[56]:


# create model : Decision Tree Regression
#----------------------------------------

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.get_params()


# In[51]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

preprocessor = StandardScaler()
modelp = make_pipeline(preprocessor, DecisionTreeRegressor(max_depth=7))


# In[53]:


modelp['decisiontreeregressor']


# In[43]:


from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=cv,
                              scoring="neg_mean_absolute_error",
                              return_train_score=True, n_jobs=2)

cv_results = pd.DataFrame(cv_results)
# cv_results


# In[17]:


# Scores of Cross validation 
#---------------------------

scores = pd.DataFrame()
scores[["train error", "test error"]] = -cv_results[["train_score", "test_score"]]

scores.plot.hist(bins=50, edgecolor="black")
plt.xlabel("Mean absolute error")
_ = plt.title("Train and test errors distribution via cross-validation")


# In[48]:


from sklearn.model_selection import validation_curve, learning_curve

max_depth = [3, 5, 7]
min_sample_leaf = [1,2,3,4,5]
min_sample_split = [3, 5, 7]
train_scores, test_scores = validation_curve(model, data, target, cv=cv,
            param_name = 'min_samples_split', param_range=min_sample_split,
            scoring='neg_mean_absolute_error', n_jobs=2)

train_errors, test_errors = -train_scores, -test_scores


# In[49]:


plt.plot(min_sample_split, train_errors.mean(axis=1), label="Training error")
plt.plot(min_sample_split, test_errors.mean(axis=1), label="Testing error")
plt.legend()

plt.xlabel("Min samples split of decision tree")
plt.ylabel("Mean absolute error")
_ = plt.title("Validation curve for decision tree")
plt.savefig('/home/nmpnguyen/conversion_model/validation_curve_decision_tree_alldata3.png')


# In[46]:


plt.plot(min_sample_leaf, train_errors.mean(axis=1), label="Training error")
plt.plot(min_sample_leaf, test_errors.mean(axis=1), label="Testing error")
plt.legend()

plt.xlabel("Min samples leaf of decision tree")
plt.ylabel("Mean absolute error")
_ = plt.title("Validation curve for decision tree")
plt.savefig('/home/nmpnguyen/conversion_model/validation_curve_decision_tree_alldata2.png')


# In[20]:


plt.plot(max_depth, train_errors.mean(axis=1), label="Training error")
plt.plot(max_depth, test_errors.mean(axis=1), label="Testing error")
plt.legend()

plt.xlabel("Maximum depth of decision tree")
plt.ylabel("Mean absolute error")
_ = plt.title("Validation curve for decision tree")
plt.savefig('/home/nmpnguyen/conversion_model/validation_curve_decision_tree_alldata.png')


# In[31]:


model = DecisionTreeRegressor(max_depth=7)

mask = np.logical_and(data[:,0]>0, target>0)
data = data[mask, :]
target = target[mask]
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.35)

model = model.fit(data_train, target_train)
data_predict = model.predict(data_test)


# In[32]:


fig, ax = plt.subplots()
h = ax.hist2d(data_test[:,0], data_predict, range=[[-10,30], [-10,80]], bins=100, norm=LogNorm())
ax.set(xlabel='355 data test', ylabel='532 data predict')
plt.colorbar(h[3], ax=ax)


fig, ax = plt.subplots()
h = ax.hist2d(data_test[:,0], target_test, range=[[-10,30], [-10,80]], bins=100, norm=LogNorm())
ax.set(xlabel='355 data test', ylabel='532 data test')
plt.colorbar(h[3], ax=ax)


# In[29]:


data_predict.shape


# --------------------------

# In[54]:


# sélectionner à apprentissage à partir des valeurs négatives 
#------------------------------------------------------------

# mask = np.logical_and(allsr355 <0, allsr532 <0)
# allsr355 = allsr355[mask]
# allsr532 = allsr532[mask]

# mat_alt = mat_alt[mask]
# mat_time = mat_time[mask]


# In[55]:


from sklearn.model_selection import train_test_split

allsr355_after = add_feature(allsr355.ravel(), mat_alt.ravel())
allsr532_after = allsr532.ravel()

mask = remove_NaN_Inf_values(allsr355_after[:,0], allsr532_after)
allsr355_after = allsr355_after[mask,:]
allsr532_after = allsr532_after[mask]
mask = np.logical_and(allsr355_after[:,0]>0, allsr532_after>0)
allsr355_after = allsr355_after[mask,:]
allsr532_after = allsr532_after[mask]

allsr355_train, allsr355_test, allsr532_train, allsr532_test = train_test_split(allsr355_after, allsr532_after, 
                                                                                test_size=0.35, random_state=0)
print(allsr355_train.shape, allsr532_test.shape)


# In[18]:


# Générer la distribution normale accumulée du dataset
#-----------------------------------------------------

import seaborn as sns 
sns.displot(pd.Series(allsr355_train[:,0]), stat='probability', kind='hist',
            kwargs={'cumulative':True, 'binrange':[-10,80]},
            kde_kws={'cumulative':True})
 


# In[56]:


from sklearn.model_selection import KFold
n_splits = 4
kf = KFold(n_splits=n_splits)

Fig, axs = plt.subplots(ncols=n_splits, figsize=(15,5))
for (id_train, id_val), (i, ax) in zip(kf.split(allsr355_train), enumerate(axs.flat)):
    XFold_train, YFold_train = allsr355_train[id_train,:], allsr532_train[id_train]
    XFold_val, YFold_val = allsr355_train[id_val,:], allsr532_train[id_val]
    print(XFold_train.shape, YFold_train.shape)
    # decision tree
    #--------------
#     sr532_pred, stat_residus, treemodel = DecisionTree_model(XFold_train, YFold_train, XFold_val, YFold_val)
#     ax.hist2d(XFold_val[:,0], sr532_pred, bins=100, range=[[-10,30], [-10, 80]], norm=LogNorm())
#     ax.set(xlabel='SR355 val_data', ylabel='SR532 predict from val_data',
#            title=f"residus = {stat_residus.loc['mean'].values} \n+/- {stat_residus.loc['std'].values}")
    #linear regression
    #--------------
#     sr532_pred, stat_residus, lr_model = LinearRgression_model(XFold_train, YFold_train, XFold_val, YFold_val)
#     ax.hist2d(XFold_val[:,0], sr532_pred, bins=100, range=[[-10,30], [-10, 80]], norm=LogNorm()) #range=[[-10,30], [-10, 80]], 
#     ax.set(xlabel='SR355 val_data', ylabel='SR532 predict from val_data',
#            title=f"residus = {stat_residus.loc['mean'].values} \n+/- {stat_residus.loc['std'].values}")
#     print(lr_model.coef_)


# In[57]:


fullsr532_predict, full_stast, treemodel_full = DecisionTree_model(allsr355_train, allsr532_train, allsr355_test, allsr532_test)
# fullsr532_predict, full_stast, lrmodel_full = LinearRgression_model(allsr355_train, allsr532_train, allsr355_test, allsr532_test)


# In[59]:


treemodel_full.get_params()


# ### Test pre-pruning of Decision Tree  

# In[62]:


from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import GridSearchCV 

param_grid = {
    "max_depth": [3,5,10,15],
#     "min_samples_split": [5,7,10],
#     "min_samples_leaf": [5,7,10],
#     "min_impurity_split": [-5.0, -1.0, 0.0]
}
clf = DTC(splitter = 'best', random_state=None)
# grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc", cv=4).fit(allsr355_train, allsr532_train)


# In[63]:


grid_cv = GridSearchCV(clf, param_grid, cv=4).fit(allsr355_train, allsr532_train)


# In[20]:


# import pickle

# pickle.dump(lrmodel_full, open('/home/nmpnguyen/conversion_model/lrmodel_atb.sav', 'wb'))
# pickle.dump(treemodel_full, open('/home/nmpnguyen/conversion_model/treemodel_sr_positive.sav', 'wb'))


# In[10]:


def scatterplot_predict_target(MLmodel, datatrain, targettrain, 
                               datatest, targettest, 
                               predicttrain, predicttest, 
                               scores, params_plots):
    ranges = params_plots['ranges']#[[-0.5,30], [-0.5, 80]]
    titles = params_plots['titles'] #y compris dataset, size of data train/test, model name
    norm = params_plots['norms']
    feature = params_plots['feature_position']
    bins = params_plots['bins']

    
    fg, axs = plt.subplots(ncols=3, nrows=2, figsize=(15,15))
    h0 = axs[0,0].hist2d(datatrain[:,feature], targettrain, bins=bins, range=ranges, norm=norm)
    axs[0,0].set(xlabel='ATB355 train_data', ylabel='ATB532 train_data',
       title=f'{titles[0]}')
    plt.colorbar(h0[3], ax=axs[0,0])

    h1 = axs[0,1].hist2d(datatest[:,feature], targettest, bins=bins, range=ranges, norm=norm)
    axs[0,1].set(xlabel='ATB355 test_data', ylabel='ATB532 test_data',
       title=f'{titles[1]}')
    plt.colorbar(h1[3], ax=axs[0,1])

    h2 = axs[0,2].hist2d(datatrain[:,feature], predicttrain, bins=bins, range=ranges, norm=norm)
    axs[0,2].set(xlabel='ATB355 train_data', ylabel='ATB532 predict from train_data',
       title=f'{titles[2]}')        
    plt.colorbar(h2[3], ax=axs[0,2])
    
    h3 = axs[1,0].hist2d(datatest[:,feature], predicttest, bins=bins, range=ranges, norm=norm)
    axs[1,0].set(xlabel='ATB355 test_data', ylabel='ATB532 predict from test_data',
       title=f'{titles[3]}')  
    plt.colorbar(h3[3], ax=axs[1,0])

    h4 = axs[1,1].hist2d(datatest[:,feature], targettest - predicttest, bins=bins, range=ranges, norm=norm)
    axs[1,1].set(xlabel='ATB355 test_data', ylabel='residus = ATB532 - ATB532pred',
       title=f'{titles[4]}')
    plt.colorbar(h4[3], ax=axs[1,1])
    
    scores['train_score'], scores['test_score'] = -scores['train_score'], -scores['test_score']
    scores = scores.reset_index()
    scores.plot(x='index', y='test_score', label='test_scores', ax=axs[1,2])
    scores.plot(x='index', y='train_score', label='train_scores', ax=axs[1,2])
    axs[1,2].legend()
    axs[1,2].set(xlabel = 'n_splits', ylabel='mean absolute error',
        title=f'{titles[5]}')
    
    plt.suptitle(f'{MLmodel}')
    return 1


# In[96]:


# plt.hist(fullsr532_predict, range=[-0.5,5])


# # TESTING WITH 3TH FEATURE

# In[6]:


# DATASET
#--------
pattern = ['/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr',
          '__xarray_dataarray_variable__']

# allsr355 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allatb355-3000-4000.nc')['__xarray_dataarray_variable__'].values
# allsr532 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allatb532-3000-4000.nc')['__xarray_dataarray_variable__'].values

allsr355 = xr.open_dataset(f'{pattern[0]}355-3000-4000.nc')[pattern[1]].values
allsr532 = xr.open_dataset(f'{pattern[0]}532-3000-4000.nc')[pattern[1]].values

# PARAMETERS
#-----------
alt = xr.open_dataset(f'{pattern[0]}355-3000-4000.nc')['range'].values
time = xr.open_dataset(f'{pattern[0]}355-3000-4000.nc')['time'].values


# In[7]:


mat_alt = np.tile(alt, (allsr355.shape[0],1))
mat_time = np.tile(time, (allsr355.shape[1], 1)).T

print(mat_alt.shape, mat_time.shape)


# In[8]:


#---------------
plt.hist2d(allsr355[allsr355>1], allsr532[allsr355>1], norm=LogNorm(), range=[[0,80], [0,80]], bins=100)
plt.colorbar()
plt.title('allsr355[allsr355>1], allsr532[allsr355>1]')


# In[13]:


allsr355_after = add_feature(allsr355.ravel(), mat_alt.ravel())
# 3e feature of X
from tqdm import tqdm
X3 = np.zeros(mat_alt.shape)
print(X3.shape)
X3[0,:] = np.nan
for j in tqdm(range(1, mat_alt.shape[1])):
    X3[:,j] = X3[:,j-1] + allsr355[:,j]*(mat_alt[:,j] - mat_alt[:,j-1])
print(X3)
allsr355_after = add_feature(allsr355_after, X3.ravel())

allsr532_after = allsr532.ravel()

mask = np.logical_and(~np.isnan(allsr355_after[:,0]), ~np.isnan(allsr355_after[:,2]), ~np.isnan(allsr532_after)) # 
# mask = remove_NaN_Inf_values(allsr355_after[:,0], allsr532_after)
allsr355_after = allsr355_after[mask,:]
allsr532_after = allsr532_after[mask]

from sklearn.model_selection import train_test_split, ShuffleSplit
cv = ShuffleSplit(test_size=0.25)
data_train, data_test, target_train, target_test = train_test_split(allsr355_after, allsr532_after, 
                                                                                test_size = 0.25, random_state=0)
print(f'shape of data train et data test {data_train.shape, data_test.shape}')



# In[11]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.semilogx(allsr355[500,:], alt, label='sr 355', zorder=10)
ax.semilogx(X3[500,:], alt, label='sr 355 integrated')
ax.legend(loc='upper right')
# ax.set_xlim(1e-1, 1e3)


# In[17]:


import pickle

allsr355_after = pd.DataFrame(allsr355_after)
allsr355_after.to_csv('/home/nmpnguyen/conversion_model/X.csv')

allsr532_after = pd.DataFrame(allsr532_after)
allsr532_after.to_csv('/home/nmpnguyen/conversion_model/Y.csv')


# In[14]:


# StandardScaler data
#--------------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

preprocessor = StandardScaler()

model = make_pipeline(preprocessor, DecisionTreeRegressor())
model.fit(data_train, target_train)

cv_results = cross_validate(model, data_train, target_train, cv=cv,
                              scoring="neg_mean_absolute_error",
                              return_train_score=True, n_jobs=2)

cv_results = pd.DataFrame(cv_results)
print(cv_results)
print(model.get_params())


# In[17]:


(-cv_results['test_score']).mean(), (-cv_results['test_score']).std()
(-cv_results['train_score']).mean(), (-cv_results['train_score']).std()


# In[14]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# training without preprocessing
#-------------------------------
model = DecisionTreeRegressor()
# max_depth=15, min_samples_split=70, min_samples_leaf=15
model.fit(data_train, target_train)

# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'max_depth' : (3, 7, 15, 30),
#     'min_samples_leaf' : (5, 10, 15, 20, 30),
# }

# model_grid_search = GridSearchCV(
#     model, param_grid=param_grid, n_jobs=2, cv=10
# )

# model_grid_search.fit(data_train, target_train)
# model_grid_search.best_param_

from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data_train, target_train, cv=cv,
                              scoring="neg_mean_absolute_error",
                              return_train_score=True, n_jobs=2)

cv_results = pd.DataFrame(cv_results)
print(cv_results)
print(model.get_params())


# In[15]:


data_predicttest = model.predict(data_test)
data_predicttrain = model.predict(data_train)
params_plot = {
    'titles' : ['1.Ipral train dataset size=0.75', '2.Test dataset size=0.25', '3.Prediction on train dataset', 
                '4.Prediction on test dataset', '5.Residus', '6.Mean absolute error'],
    'bins' : 100,
    'norms' : LogNorm(),
    'ranges': [[-10,30], [-10,80]],
    'feature_position':0
}
# scatterplot_predict_target(model, data_train, target_train, data_test, target_test,
#                           data_predicttrain, data_predicttest, cv_results, params_plot)


# In[18]:


pd.DataFrame(data_train).to_pickle(Path('/home/nmpnguyen/conversion_model/comparaison', 'ipral_2018_learned_train_dataset.pkl'))
pd.DataFrame(data_test).to_pickle(Path('/home/nmpnguyen/conversion_model/comparaison', 'ipral_2018_learned_TEST_dataset.pkl'))
pd.DataFrame(target_train).to_pickle(Path('/home/nmpnguyen/conversion_model/comparaison', 'ipral_2018_learned_traintarget_dataset.pkl'))
pd.DataFrame(target_test).to_pickle(Path('/home/nmpnguyen/conversion_model/comparaison', 'ipral_2018_learned_TESTtarget_dataset.pkl'))
pd.DataFrame(data_predicttest).to_pickle(Path('/home/nmpnguyen/conversion_model/comparaison', 'ipral_2018_learned_TESTpredict_dataset.pkl'))


# In[109]:


pd_classed = pd.DataFrame(np.concatenate([data_test, target_test.reshape(-1,1), data_predicttest.reshape(-1,1)], axis=1), 
                          columns = ['sr355', 'alt', 'sr532', 'sr532_predicted'])
pd_classed['absolute_error'] = (pd_classed['sr532'] - pd_classed['sr532_predicted'])

pd_classed

pd_classed['mesures_range'] = pd.cut(pd_classed['sr355'], pd.interval_range(start=0, end=60, freq=1))
pd_mean = pd_classed.groupby('mesures_range').agg({'absolute_error': lambda x: x.mean(skipna=True)})['absolute_error']
pd_std = pd_classed.groupby('mesures_range').agg({'absolute_error': lambda x: x.std(skipna=True)})['absolute_error']

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(np.arange(0,60,1), pd_mean, 'b-')
ax.fill_between(np.arange(0,60,1), pd_mean+pd_std, pd_mean-pd_std, color='b', alpha=0.2)
ax.set(xlabel='SR355', ylabel='<e> +/- std(e)')


# In[111]:


pd_mean = pd_classed.groupby('alt').agg({'absolute_error': lambda x: x.mean(skipna=True)})['absolute_error']
pd_std = pd_classed.groupby('alt').agg({'absolute_error': lambda x: x.std(skipna=True)})['absolute_error']


fig, ax = plt.subplots()
ax.plot(pd_mean.reset_index()['alt'], pd_mean, 'b-')
ax.fill_between(pd_mean.reset_index()['alt'], pd_mean+pd_std, pd_mean-pd_std, color='b', alpha=0.2)
ax.set(xlabel='Altitude', ylabel='<e> +/- std(e)')
# ax.set_xlim(0,15000)


# In[112]:


pd_classed['mesures_range'] = pd.cut(pd_classed['sr355_integrated'], pd.interval_range(start=0, end=np.nanmax(pd_classed['sr355_integrated']), freq=20))
pd_mean = pd_classed.groupby('mesures_range').agg({'absolute_error': lambda x: x.mean(skipna=True)})['absolute_error']
pd_std = pd_classed.groupby('mesures_range').agg({'absolute_error': lambda x: x.std(skipna=True)})['absolute_error']
# pd_mean
fig, ax = plt.subplots()
ax.plot(np.arange(0,np.nanmax(pd_classed['sr355_integrated']),20)[:-1], pd_mean, 'b-')
ax.fill_between(np.arange(0,np.nanmax(pd_classed['sr355_integrated']),20)[:-1], pd_mean+pd_std, pd_mean-pd_std, color='b', alpha=0.2)
ax.set(xlabel='SR355 integrated', ylabel='<e> +/- std(e)')


# In[ ]:


fig, ax = plt.subplots()
ax.hist2d(data_predicttest, target_test, range=[[-10, 80], [-10, 80]], bins=100, norm=LogNorm())
ax.set(xlabel='SR532 predict from data_test', ylabel='SR532 data_test')


# In[26]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
scores = mean_absolute_error(data_predicttest, target_test)
print(f'mean_absolute_error : {mean_absolute_error(data_predicttest, target_test)}')
print(f'mean_squared_error : {mean_squared_error(data_predicttest, target_test)}')
print(f'r2_score : {r2_score(data_predicttest, target_test)}')


# In[132]:


import pickle

pickle.dump(model, open('/home/nmpnguyen/conversion_model/tree_3f.sav', 'wb'))
# pickle.load(open('/home/nmpnguyen/conversion_model/tree_3f.sav', 'rb'))


# #### tree model default
# 
#     {'ccp_alpha': 0.0,
#      'criterion': 'mse',
#      'max_depth': None,
#      'max_features': None,
#      'max_leaf_nodes': None,
#      'min_impurity_decrease': 0.0,
#      'min_impurity_split': None,
#      'min_samples_leaf': 1,
#      'min_samples_split': 2,
#      'min_weight_fraction_leaf': 0.0,
#      'presort': 'deprecated',
#      'random_state': None,
#      'splitter': 'best'}

# In[101]:



# training with preprocessing
#-------------------------------
modelp = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=15))
modelp.fit(data_train, target_train)

cv_results_p = cross_validate(modelp, data_train, target_train, cv=cv,
                              scoring="neg_mean_absolute_error",
                              return_train_score=True, n_jobs=2)

cv_results_p = pd.DataFrame(cv_results_p)
print(cv_results_p)


data_predicttest = modelp.predict(data_test)
data_predicttrain = modelp.predict(data_train)
params_plot = {
    'titles' : ['1.Ipral train dataset size=0.75', '2.Test dataset size=0.25', '3.Prediction on train dataset', 
                '4.Prediction on test dataset', '5.Residus', '6.Mean absolute error'],
    'bins' : 100,
    'norms' : LogNorm(),
    'ranges': [[-10,30], [-10,80]],
    'feature_position':0
}
scatterplot_predict_target(modelp, data_train, target_train, data_test, target_test,
                          data_predicttrain, data_predicttest, cv_results, params_plot)


# In[128]:


# find best params with 3 features 

print(f'{model}')
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": np.arange(1, 10, 2)}
tree_reg = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid)
tree_reg.fit(data_train, target_train)


# In[129]:


tree_reg.best_params_['max_depth']


# In[13]:


from sklearn.model_selection import KFold
n_splits = 4
kf = KFold(n_splits=n_splits)

Fig, axs = plt.subplots(ncols=n_splits, figsize=(15,5))
for (id_train, id_val), (i, ax) in zip(kf.split(allsr355_train), enumerate(axs.flat)):
    XFold_train, YFold_train = allsr355_train[id_train,:], allsr532_train[id_train]
    XFold_val, YFold_val = allsr355_train[id_val,:], allsr532_train[id_val]
    print(XFold_train.shape, YFold_train.shape)
    sr532_pred, stat_residus, treemodel3 = DecisionTree_model(XFold_train, YFold_train, XFold_val, YFold_val)
    #plot
    ax.hist2d(XFold_val[:,0], sr532_pred, bins=100, range=[[-10,30], [-10, 80]], norm=LogNorm())
    ax.set(xlabel='SR355 val_data', ylabel='SR532 predict from val_data',
           title=f"residus = {stat_residus.loc['mean'].values} \n+/- {stat_residus.loc['std'].values}")


# In[20]:



fullsr532_predict3, full_stast3, treemodel_full3 = DecisionTree_model(allsr355_train, allsr532_train, allsr355_test, allsr532_test)
fg, ax = plt.subplots()
ax.hist2d(allsr355_test[:,0], fullsr532_predict3, bins=100, range=[[-10,30], [-10, 80]], norm=LogNorm())
ax.set(xlabel='SR355 val_data', ylabel='SR532 predict from val_data',
       title=f"residus = {full_stast3.loc['mean'].values} \n+/- {full_stast3.loc['std'].values}")


# In[102]:


### SR532 to SR355 
from sklearn.model_selection import train_test_split

allsr355_after = allsr355.ravel()
allsr532_after = add_feature(allsr532.ravel(), mat_alt.ravel())

mask = remove_NaN_Inf_values(allsr355_after, allsr532_after[:,0])
allsr355_after = allsr355_after[mask]
allsr532_after = allsr532_after[mask,:]

allsr532_train, allsr532_test, allsr355_train, allsr355_test= train_test_split(allsr532_after, allsr355_after,  
                                                                                test_size=0.35, random_state=0)


# In[106]:


fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(15,5))
ax.hist2d(allsr355_train, allsr532_train[:,0], bins=100, range=[[-10,30], [-10, 80]], norm=LogNorm())
ax2.hist2d(allsr355_test, allsr532_test[:,0], bins=100, range=[[-10,30], [-10, 80]], norm=LogNorm())


# In[103]:


from sklearn.model_selection import KFold
n_splits = 4
kf = KFold(n_splits=n_splits)

Fig, axs = plt.subplots(ncols=n_splits, figsize=(15,5))
for (id_train, id_val), (i, ax) in zip(kf.split(allsr532_train), enumerate(axs.flat)):
    XFold_train, YFold_train = allsr532_train[id_train,:], allsr355_train[id_train]
    XFold_val, YFold_val = allsr532_train[id_val,:], allsr355_train[id_val]
    print(XFold_train.shape, YFold_train.shape)
    sr355_pred, stat_residus, treemodel = DecisionTree_model(XFold_train, YFold_train, XFold_val, YFold_val)
    #plot
    ax.hist2d(sr355_pred, XFold_val[:,0], bins=100, range=[[-10,30], [-10, 80]], norm=LogNorm())
    ax.set(ylabel='SR532 val_data', xlabel='SR355 predict from val_data',
           title=f"residus = {stat_residus.loc['mean'].values} \n+/- {stat_residus.loc['std'].values}")


# In[104]:


fullsr355_predict, full_stast, treemodel_full = DecisionTree_model(allsr532_train, allsr355_train, allsr532_test, allsr355_test)


# In[105]:


fg, ax = plt.subplots()
ax.hist2d(fullsr355_predict, allsr532_test[:,0], bins=100, range=[[-10,30], [-10, 80]], norm=LogNorm())
ax.set(ylabel='SR532 test_data', xlabel='SR355 predict from test_data',
       title=f"residus = {full_stast.loc['mean'].values} \n+/- {full_stast.loc['std'].values}")


# ### EXEMPLE 1 QUICKLOOK IPRAL

# In[44]:


### essayer avec un cas d'étude en quicklook 
dt = xr.open_dataset('/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/2020/ipral_1a_Lz1R15mF30sPbck_v01_20200904_000000_1440.nc')

dtsr355 = (dt['calibrated']/dt['simulated']).sel(wavelength=355).where(dt['flags'].sel(wavelength=355) == 0, drop=False).values 
dtsr532 = (dt['calibrated']/dt['simulated']).sel(wavelength=532).where(dt['flags'].sel(wavelength=532) == 0, drop=False).values
dtalt = dt['range'].values
dttime = dt['time'].values

negative_where = np.where(dtsr355 < 0)
dtsr355[negative_where] = np.nan

import matplotlib.dates as mdates
fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(15,5))
cmap = plt.cm.turbo
cmap.set_under('lightgrey')
# 532nm original
p = ax.pcolormesh(dttime, dtalt, dtsr532.T, cmap='turbo', vmin=0, vmax=20)
plt.colorbar(p, ax=ax, label='sr532', extend='both')
ax.set_ylim(0,14000)
ax.set(xlabel='time', ylabel='range', title='sr532')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# 355nm original
p = ax2.pcolormesh(dttime, dtalt, dtsr355.T, cmap='turbo', vmin=0, vmax=20)
plt.colorbar(p, ax=ax2, label='sr355', extend='both')
ax2.set_ylim(0,14000)
ax2.set(xlabel='time', ylabel='range', title='sr355')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


# In[47]:


### essayer avec un cas d'étude en quicklook (suite)
dtX2 = add_feature(dtsr355.ravel(), np.tile(dtalt, dtsr355.shape[0]))
dtalt_mat = np.tile(dtalt, (dtsr355.shape[0],1))
# 3e feature of X_mat
from tqdm import tqdm
X3 = np.zeros(dtalt_mat.shape)
print(X3.shape)
# X3[0,:] = np.nan
for j in tqdm(range(1, dtalt_mat.shape[1])):
    X3[:,j] = X3[:,j-1] + dtsr355[:,j]*(dtalt_mat[:,j] - dtalt_mat[:,j-1])

dtX = add_feature(dtX2, X3.ravel())


# In[55]:


dtX[np.where(~np.isnan(dtX))[0],:]


# In[49]:


indx = np.where(np.logical_and(~np.isnan(dtX), ~np.isinf(dtX)))
print(f'Index : {indx} and {indx[0].shape}')
dtX_input = dtX[np.unique(indx[0]), :]


dtXpredict = model.predict(dtX_input)
tmp = np.full(dtsr532.shape[0], np.nan)
dtXpredict = dtXpredict.reshape(dtsr532.shape)

# fig, ax = plt.subplots()
# cmap = plt.cm.turbo
# cmap.set_under('lightgrey')

# p = ax.pcolormesh(dttime, dtalt, dtXpredict.T, cmap=cmap, vmin=0, vmax=20)
# plt.colorbar(p, ax=ax, label='sr532 predict', extend='both')
# ax.set_ylim(4000,14000)
# ax.set(xlabel='time', ylabel='range', title='sr532 predict')
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


# In[30]:


print(dtXpredict[negative_where], dtsr532)


# # Apply on ER2

# ------------------

# In[7]:


allsr532 = xr.open_dataset('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/HSRL2_ER2_allsr532_v3.nc')
allsr532 = allsr532.assign_coords(time = pd.to_datetime(allsr532['time']))['__xarray_dataarray_variable__']
# allsr532 = allsr532.resample(time = '15min').mean(skipna=True)


# In[8]:


allsr355 = xr.open_dataset('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/HSRL2_ER2_allsr355_v3.nc')
allsr355 = allsr355.assign_coords(time = pd.to_datetime(allsr355['time']))['__xarray_dataarray_variable__']
# allsr355 = allsr355.resample(time = '15min').mean(skipna=True)


# In[9]:


mat_alt = np.tile(allsr355['altitude'].values, (allsr355.values.shape[0], 1))
mat_time = np.tile(allsr355['time'].values, (allsr355.values.shape[1], 1))


# ---------------

# In[60]:


mat_alt.shape, allsr355.shape


# In[13]:


from sklearn.model_selection import train_test_split
# allsr355 = allsr355.values
# allsr532 = allsr532.values

allsr355_after = add_feature(allsr355.ravel(), mat_alt.ravel())
allsr532_after = allsr532.ravel()

mask = np.logical_and(~np.isnan(allsr355_after[:,0]), ~np.isnan(allsr532_after))
allsr355_after = allsr355_after[mask,:]
allsr532_after = allsr532_after[mask]

allsr355_train, allsr355_test, allsr532_train, allsr532_test = train_test_split(allsr355_after, allsr532_after, 
                                                                                test_size=0.35, random_state=0)
print(allsr355_train.shape, allsr532_test.shape)


# In[62]:


n_splits = 3
kf = KFold(n_splits=n_splits)

Fig, axs = plt.subplots(ncols=n_splits, figsize=(15,5))
for (id_train, id_val), (i, ax) in zip(kf.split(allsr355_train), enumerate(axs.flat)):
    XFold_train, YFold_train = allsr355_train[id_train,:], allsr532_train[id_train]
    XFold_val, YFold_val = allsr355_train[id_val,:], allsr532_train[id_val]
    print(XFold_train.shape, YFold_train.shape)
    sr532_pred, stat_residus, treemodel_for_er2 = DecisionTree_model(XFold_train, YFold_train, XFold_val, YFold_val)
    #plot
    ax.hist2d(XFold_val[:,0], sr532_pred, bins=100, range=[[-10,30], [-10, 80]], norm=LogNorm())
    ax.set(xlabel='SR355 val_data', ylabel='SR532 predict from val_data',
           title=f"ER2 residus = {stat_residus.loc['mean'].values} \n+/- {stat_residus.loc['std'].values}")


# In[15]:


fullsr532_predict_er2, full_stast_er2, treemodel_full_er2 = DecisionTree_model(allsr355_train, allsr532_train, allsr355_test, allsr532_test)


# In[18]:


# clf = treemodel_full_er2.fit(allsr355_train, allsr532_train)
from sklearn import tree
plotree, ax = plt.subplots(figsize=(10,8))
treemodel_full_er2 = treemodel_full_er2.fit(allsr355_train, allsr532_train)
tree.plot_tree(treemodel_full_er2, max_depth=2, filled=False, ax=ax)
plt.title("Decision tree trained on 2 the ER2 features: SR355 vs z")
plt.savefig('/homedata/nmpnguyen/ORACLES-ER2/Figs/decision_tree_tmp.png')


# In[101]:


fg, (ax, ax2) = plt.subplots(ncols=2, figsize=(15,5))
h = ax.hist2d(allsr355_test[:,0], fullsr532_predict_er2, bins=[100, 100], range=[[-1,30], [-1, 80]], norm=LogNorm(vmax=1e5))
ax.set(ylabel='SR532 test_data', xlabel='SR355 predict from test_data',
       title=f"ER2 residus = {full_stast_er2.loc['mean'].values} \n+/- {full_stast_er2.loc['std'].values}")
plt.colorbar(h[3], ax=ax, label='counts')

h2 = ax2.hist2d(allsr355_test[:,0], allsr532_test, bins=[100, 100], range=[[-1,30], [-1, 80]], norm=LogNorm(vmax=1e5))
ax2.set(ylabel='SR532 test_data', xlabel='SR355 test_data')
plt.colorbar(h2[3], ax=ax2, label='counts')


# ### EXEMPLE 1 PROFIL OU 1 QUICKLOOK DE ER2

# In[92]:


dt = xr.open_dataset('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/HSRL2_ER2_20160819_R8.h5')

dtsr355 = (dt['calibrated']/dt['molecular']).sel(wavelength=355).values
dtsr532 = (dt['calibrated']/dt['molecular']).sel(wavelength=532).values
dtalt = dt['altitude'].values

dtX = add_feature(dtsr355.ravel(), np.tile(dtalt, dtsr355.shape[0]))


import matplotlib.dates as mdates
fig, ax = plt.subplots()
p = ax.pcolormesh(dt['time'].values, dtalt, dtsr532.T, cmap='turbo', vmin=0, vmax=10)
plt.colorbar(p, ax=ax, label='sr355', extend='both')
ax.set(xlabel='time', ylabel='range', title='sr532 test_data')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


# In[90]:


# dtX = dtX[~np.isnan(dtX[:,0]),:]
dtYpred_er2= treemodel_full_er2.predict(np.nan_to_num(dtX))
dtYpred_er2 = dtYpred_er2.reshape(dtsr355.shape)

fig, ax = plt.subplots()
cmap = plt.cm.turbo 
cmap.set_under('lightgrey')
p = ax.pcolormesh(dt['time'].values, dtalt, dtYpred_er2.T, cmap=cmap, vmin=0.2, vmax=10)
plt.colorbar(p, ax=ax, label='sr355', extend='max')
ax.set(xlabel='time', ylabel='range', title='sr532 predict')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


# ### Build data train > Decision Tree Model
# ---------------------------------------------------------

# In[157]:


# get profil from day file
# get get data from profil 
# get all moments 

def get_profil_random_from_dayfile(daytime, nb_profils):
    daytime = pd.to_datetime(daytime)
    import random 
    ns = random.sample(range(len(daytime)), min(nb_profils, len(daytime)))   
    return ns

def get_data_from_profil(dayfile, z_limite, nb_profils):
    z = dayfile['range'].values
    z_selected = z[z < z_limite] if isinstance(z_limite,int) else z[(z > z_limite[0])&(z < z_limite[1])]
    x355 = dayfile['Total_ScattRatio_355'].where(dayfile['flags_355']==0, drop=False).sel(range=z_selected)#
    x532 = dayfile['Total_ScattRatio_532'].where(dayfile['flags_532']==0, drop=False).sel(range=z_selected)#.isel(time=profils_selected).values
    x355 = x355.resample(time = '15min').mean(skipna=True)
    x532 = x532.resample(time = '15min').mean(skipna=True)
    times = np.intersect1d(x355.time, x532.time)
    # get_profil_random_from_dayfile 
    profils_selected = get_profil_random_from_dayfile(times, nb_profils)
    x355 = x355.isel(time=profils_selected).values
    x532 = x532.isel(time=profils_selected).values
    # remove nan values
    mask_nan = ~np.isnan(x355)&~np.isnan(x532)
    x355 = x355[mask_nan]
    x532 = x532[mask_nan]
    z2D_selected = np.tile(z_selected, (len(profils_selected), 1))[mask_nan]
    return x355, z2D_selected, x532

def get_all_data_selected_2features(alldaysfile, nb_profils, z_limite):
    all_X355 = np.array([[],[]])
    all_X532 = np.array([])
    for day in alldaysfile:
        day_data = xr.open_dataset(day)
#         profils_id = get_profil_random_from_dayfile(day_data, nb_profils)
        X355, Zselected, X532 = get_data_from_profil(day_data, z_limite, nb_profils)
        all_X355 = np.concatenate([all_X355, [X355, Zselected]], axis=1)
        all_X532 = np.concatenate([all_X532, X532])
    return np.array(all_X355).T, np.array(all_X532)




# In[432]:


Listfiles = sorted(Path('/homedata/nmpnguyen/IPRAL/NETCDF/v2/').glob('ipral_calib_03_2018*.nc'))
X355_Z, X532= get_all_data_selected_2features(Listfiles, 100, 14000)
# X355_Z, X532= get_all_data_selected_3features(Listfiles[:100], 100, 14000)


# In[433]:


X355_Z.shape, X532.shape


# In[434]:


get_ipython().magic('matplotlib inline')
fig, ax = plt.subplots()
h = ax.hist2d(X355_Z[:,0],  X532, bins=100, range=[[0, 40],[0, 80]], norm=LogNorm())
plt.colorbar(h[3], ax=ax)


# In[403]:


Y = X532[np.logical_and(X355_Z[:,0]>0,X532>0)]# training
X = X355_Z[np.logical_and(X355_Z[:,0]>0,X532>0),:]# training
# Y = Y/X[:,0]# training

# Y=Y[np.logical_and(X[:,1]<14000,X[:,1]>3000)] # training
# X = X[np.logical_and(X[:,1]<14000,X[:,1]>3000)] # training

# Y=Y[X[:,0]>1.2]
# X = X[X[:,0]>1.2,:]

X.shape


# In[404]:


fig, ax = plt.subplots()
ax.hist(Y[Y<8], bins=10)


# In[405]:


from sklearn.tree import DecisionTreeRegressor as DTR
tree = DTR(min_samples_leaf=5)
tree.fit(X,Y)


# In[408]:


file_test = Path('/homedata/nmpnguyen/IPRAL/NETCDF/v2/ipral_calib_03_20181017_000000_1440.nc')
Xtest, Ytest = get_all_data_selected_2features([file_test], 1, 14000)
# Xtest, Ytest = get_all_data_selected_3features([file_test], 100, 14000)


# In[409]:


Yt = Ytest[np.logical_and(Xtest[:,0]>0,Ytest>0)]
Xt = Xtest[np.logical_and(Xtest[:,0]>0,Ytest>0),:]
# Yt = Yt/Xt[:,0]

# Yt=Yt[np.logical_and(Xt[:,1]<14000,Xt[:,1]>3000)]
# Xt = Xt[np.logical_and(Xt[:,1]<14000,Xt[:,1]>3000)]

# Yt = Yt[Xt[:,0] >1.2]
# Xt = Xt[Xt[:,0]>1.2, :]


Xt.shape
Y_pred = tree.predict(Xt)


# In[377]:


fig, ax = plt.subplots()

Yo = Ytest[np.logical_and(Xtest[:,0]>0,Ytest>0)]
Xo = Xtest[np.logical_and(Xtest[:,0]>0,Ytest>0),:]


# ax.plot(Ytest[np.logical_and(Xtest[:,0]>0,Ytest>0)], Xt[:,1], color='b')
ax.plot(Yo, Xo[:,1], color='b')
ax.plot(Y_pred, Xt[:,1], color='r')
# ax.plot(Xt[:,0], Xt[:,1], color='g')
ax.set_ylim(0, 4000)


# In[410]:


# list of test days
list_files_test = sorted(Path('/homedata/nmpnguyen/IPRAL/NETCDF/v2/').glob('ipral_calib_03_20180*.nc'))
Xtest, Ytest = get_all_data_selected_2features(list_files_test, 100, 14000)
Yt = Ytest[np.logical_and(Xtest[:,0]>0,Ytest>0)]
Xt = Xtest[np.logical_and(Xtest[:,0]>0,Ytest>0),:]


# In[411]:


# scatterplot before applying model
fig, ax = plt.subplots()
h = ax.hist2d(Xt[:,0],  Yt, bins=100, range=[[-1, 40],[-1, 80]], norm=LogNorm())
plt.colorbar(h[3], ax=ax)
ax.set(xlabel='sr355_mesured', ylabel='sr532_mesured')


# In[245]:


alsr355 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['__xarray_dataarray_variable__'].values
alsr532 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr532-3000-4000.nc')['__xarray_dataarray_variable__'].values

# fig, ax = plt.subplots()
# pcm = ax.hist2d(alsr355.ravel(), alsr532.ravel(), bins=100, range=[[-10,40], [-10,80]], norm=LogNorm(vmin=1, vmax=1e5))
# plt.colorbar(pcm[3], ax=ax)


# In[171]:


t = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['time']
list_random50 = get_profil_random_from_dayfile(t, np.int(alsr532.shape[0]/2))
list_notrandom50 = list(set(range(alsr532.shape[0])) - set(list_random50))

print(type(list_notrandom50))


# In[161]:


X = alsr355[list_random50, :]
Y = alsr532[list_random50, :]
mat_alt = np.tile(xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['range'].values, (X.shape[0], 1))
print(mat_alt.ravel().shape)
print(X.shape)
X = np.vstack((X.ravel(), mat_alt.ravel()))
X = X.T
Y = Y.ravel()
print(X.shape)



# In[162]:


Yinput = Y[np.logical_and(~np.isnan(X[:,0]), ~np.isnan(Y))]
Xinput = X[np.logical_and(~np.isnan(X[:,0]), ~np.isnan(Y)),:]

# generate Decision Tree model
from sklearn.tree import DecisionTreeRegressor as DTR
tree = DTR(min_samples_leaf=5)
tree.fit(Xinput,Yinput)


# In[114]:


# print(X[:,0], Y)
# fig, ax = plt.subplots()
# pcm = ax.hist2d(Xinput[:,0], Yinput, bins=100, range=[[-10,40], [-10,80]], norm=LogNorm(vmin=1, vmax=1e5))
# plt.colorbar(pcm[3], ax=ax)
# ax.set(xlabel='SR355_mesured', ylabel='SR532_mesured')
# fig.savefig(f'/home/nmpnguyen/scatter_train_data_{len(Xinput[:,0])}.png')


# In[163]:


# generate Linear Regression
slope, intercept = [4.11964, -1.7804]

# generate Polynomial Regresssion
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_features = PolynomialFeatures(degree=2)
X_features = poly_features.fit_transform(Xinput)
reg = LinearRegression()
reg.fit(X_features, Yinput)


# In[246]:


# generate Data Test 

X = alsr355[list_notrandom50, :]
Y = alsr532[list_notrandom50, :]
mat_alt = np.tile(xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['range'].values, (X.shape[0], 1))
X = np.vstack((X.ravel(), mat_alt.ravel()))
X = X.T
Y = Y.ravel()
Xtests = X[np.logical_and(~np.isnan(X[:,0]), ~np.isnan(Y)),:]
Ytests = Y[np.logical_and(~np.isnan(X[:,0]), ~np.isnan(Y))]


# In[247]:


# create y predict from DCT
Y_pred = tree.predict(Xtests)
print(Xtests.shape, Y_pred.shape, Ytests.shape)
residus_y = Ytests - Y_pred
stat_DCT = pd.DataFrame(residus_y).describe()
print(stat_DCT)


# In[248]:


# create y predict from Linear Regression
Y_pred2 = slope*Xtests[:,0] + intercept
print(Xtests.shape, Y_pred.shape, Ytests.shape)
residus2_y = Ytests - Y_pred2
stat_LR = pd.DataFrame(residus2_y).describe()
print(stat_LR)


# In[249]:


# create y predict from Polynomial Regression
Xtests_features = poly_features.fit_transform(Xtests)

Y_pred3 = reg.predict(Xtests_features)
# print(Xtests.shape, Y_pred.shape, Ytests.shape)
residus3_y = Ytests - Y_pred3
stat_PR = pd.DataFrame(residus3_y).describe()
print(stat_PR)


# In[175]:


fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(11,5))
pcm = ax.hist2d(Xinput[:,0], Yinput, bins=100, range=[[-10,40], [-10,80]], norm=LogNorm(vmin=1, vmax=1e5))
plt.colorbar(pcm[3], ax=ax)
ax.set(xlabel='SR355_mesured', ylabel='SR532_mesured', title=f'Input Train {len(Xinput[:,0])}')

pcm = ax2.hist2d(Xtests[:,0], Ytests, bins=100, range=[[-10,40], [-10,80]], norm=LogNorm(vmin=1, vmax=1e5))
plt.colorbar(pcm[3], ax=ax2)
ax2.set(xlabel='SR355_test', ylabel='SR532_test', title=f'Input Test {len(Xtests[:,0])}')
fig.savefig(f'/home/nmpnguyen/scatter_test_data_{len(Xtests[:,0])}_trainby{len(Xinput[:,0])}.png')


# In[251]:


fig, (ax, ax2, ax3) = plt.subplots(ncols=3, figsize=(17,5)) #from DCTree
pcm = ax.hist2d(Xtests[:,0], Y_pred, bins=100, range=[[-10,40], [-10,80]], norm=LogNorm(vmin=1, vmax=1e5))
plt.colorbar(pcm[3], ax=ax)
ax.set(xlabel='SR355_mesured', ylabel='SR532_predict', title=f'Output DCT {len(Xtests[:,0])}')
# fig.savefig(f'/home/nmpnguyen/scatter_predictDTC_data_{len(Xtests[:,0])}.png')

pcm = ax2.hist2d(Xtests[:,0], Y_pred2, bins=100, range=[[-10,20], [-10,60]], norm=LogNorm(vmin=1, vmax=1e5))
plt.colorbar(pcm[3], ax=ax2)
ax2.set(xlabel='SR355_mesured', ylabel='SR532_predict', title=f'Output LR {len(Xtests[:,0])}')

pcm = ax3.hist2d(Xtests[:,0], Y_pred3, bins=100, range=[[-10,20], [-10,60]], norm=LogNorm(vmin=1, vmax=1e5))
plt.colorbar(pcm[3], ax=ax3)
ax3.set(xlabel='SR355_mesured', ylabel='SR532_predict', title=f'Output PolyR {len(Xtests[:,0])}')

fig.savefig(f'/home/nmpnguyen/scatter_predict_data_{len(Xtests[:,0])}_trainby{len(Xinput[:,0])}.png')


# In[333]:


fig, (ax, ax2, ax3) = plt.subplots(ncols=3, figsize=(17,5)) #from DCTree
pcm = ax.hist2d(Ytests, Y_pred, bins=100, range=[[-10,80], [-10,80]], norm=LogNorm(vmin=1, vmax=1e5))
plt.colorbar(pcm[3], ax=ax)
ax.set(ylabel='SR532_predict', xlabel='SR532_measured', title=f'Output DCT {len(Xtests[:,0])}')
# fig.savefig(f'/home/nmpnguyen/scatter_predictDTC_data_{len(Xtests[:,0])}.png')

pcm = ax2.hist2d(Ytests, Y_pred2, bins=100, range=[[-10,80], [-10,80]], norm=LogNorm(vmin=1, vmax=1e5))
plt.colorbar(pcm[3], ax=ax2)
ax2.set(ylabel='SR532_predict', xlabel='SR532_measured', title=f'Output LR {len(Xtests[:,0])}')

pcm = ax3.hist2d(Ytests, Y_pred3, bins=100, range=[[-10,80], [-10,80]], norm=LogNorm(vmin=1, vmax=1e5))
plt.colorbar(pcm[3], ax=ax3)
ax3.set(ylabel='SR532_predict', xlabel='SR532_measured', title=f'Output PolyR {len(Xtests[:,0])}')

fig.savefig(f'/home/nmpnguyen/scatter_532predict_532measured_{len(Xtests[:,0])}_trainby{len(Xinput[:,0])}.png')


# In[112]:


# fig, ax = plt.subplots() #from Linear Regression
# pcm = ax.hist2d(Xtests[:,0], Y_pred2, bins=100, range=[[-10,20], [-10,60]], norm=LogNorm(vmin=1, vmax=1e5))
# plt.colorbar(pcm[3], ax=ax)
# ax.set(xlabel='SR355_mesured', ylabel='SR532_predict')
# fig.savefig(f'/home/nmpnguyen/scatter_predictLR_data_{len(Xtests[:,0])}.png')


# In[113]:


# fig, ax = plt.subplots() #from Linear Regression
# pcm = ax.hist2d(Xtests[:,0], Y_pred3, bins=100, range=[[-10,20], [-10,60]], norm=LogNorm(vmin=1, vmax=1e5))
# plt.colorbar(pcm[3], ax=ax)
# ax.set(xlabel='SR355_mesured', ylabel='SR532_predict')
# fig.savefig(f'/home/nmpnguyen/scatter_predictPolyR_data_{len(Xtests[:,0])}.png')


# In[184]:


import random

# generate Data Test 1 Profile

# d = xr.open_dataset('/homedata/nmpnguyen/IPRAL/RF/Calibrated/zone-3000-4000/ipral_1a_Lz1R15mF30sPbck_v01_20181106_000000_1440.nc')
# X = (d['calibrated']/d['simulated']).sel(wavelength=355).resample(time='15min').mean(skipna=True)
# Y = (d['calibrated']/d['simulated']).sel(wavelength=532).resample(time='15min').mean(skipna=True)
# mat_alt = d['range'].values

mat_alt = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['range'].values
# i = random.randint(0, alsr355.shape[0])
i = random.choice(list_random50)

X = alsr355[i,:]
Y = alsr532[i,:]
# X = X.values[i,:]
# Y = Y.values[i,:]
get_ipython().magic('matplotlib inline')
f, ax = plt.subplots()
ax.plot(X[mat_alt<14000], mat_alt[mat_alt<14000], color='b')
ax.plot(Y[mat_alt<14000], mat_alt[mat_alt<14000], color='g')
# ax.set_xlim(-1,10)

print(X.shape, Y.shape, mat_alt.shape)
X = np.vstack((X[mat_alt<14000], mat_alt[mat_alt<14000]))
X = X.T
Y = Y[mat_alt<14000]
Xtests = X[np.logical_and(~np.isnan(X[:,0]), ~np.isnan(Y))]
Ytests = Y[np.logical_and(~np.isnan(X[:,0]), ~np.isnan(Y))]
print(Xtests.shape, Ytests.shape)

# create y predict
Y_pred = tree.predict(Xtests)
Y_pred2 = slope*Xtests[:,0] + intercept
Xtests_features = poly_features.fit_transform(Xtests)
Y_pred3 = reg.predict(Xtests_features)



# In[185]:


get_ipython().magic('matplotlib inline')
fig, (ax, ax2, ax3) = plt.subplots(ncols=3, figsize=(17,5))
title=xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['time'].values[i]

ax.plot(X[:,0], X[:,1], color='b')
ax.plot(Y, X[:,1], color='g')
ax.plot(Y_pred, Xtests[:,1], color='r')
ax.set(title=f'Output DCT {title}')
# fig.savefig(f'/home/nmpnguyen/profile_predictDCT_trainby{len(Xinput[:,0])}_{title}.png')


ax2.plot(X[:,0], X[:,1], color='b')
ax2.plot(Y, X[:,1], color='g')
ax2.plot(Y_pred2, Xtests[:,1], color='r')
ax2.set(title=f'Output LR')
# fig.savefig(f'/home/nmpnguyen/profile_predictLR_trainby{len(Xinput[:,0])}_{title}.png')

ax3.plot(X[:,0], X[:,1], color='b', label='355')
ax3.plot(Y, X[:,1], color='g', label='532')
ax3.plot(Y_pred3, Xtests[:,1], color='r', label='532_predict')
ax3.set(title=f'Output PolyR')
ax3.legend()
fig.savefig(f'/home/nmpnguyen/profile_predict_trainby{len(Xinput[:,0])}_{title}.png')


# In[186]:


# fig, ax = plt.subplots()
# ax.plot(X[:,0], X[:,1], color='b')
# ax.plot(Y, X[:,1], color='g')
# # ax.set_ylim(0,14000)
# ax.plot(Y_pred3, Xtests[:,1], color='r')
# # ax.set_xlim(-1,10)
# ax.set(title=f'{title}')
# fig.savefig(f'/home/nmpnguyen/profile_predictPolyR_trainby{len(Xinput[:,0])}_{title}.png')


# In[351]:


# Generate Data Test 1 Jour
# random.choice(np.unique(pd.to_datetime(t).strftime('%Y%m%d')))
alsr355 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['__xarray_dataarray_variable__']
alsr532 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr532-3000-4000.nc')['__xarray_dataarray_variable__']
mat_alt = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['range'].values

# ii = np.where(pd.to_datetime(t).strftime('%Y%m%d') == random.choice(np.unique(pd.to_datetime(t).strftime('%Y%m%d'))))[0]

X = alsr355[ii,:]
Y = alsr532[ii,:]

print(X.shape, Y.shape)

get_ipython().magic('matplotlib inline')
fig, ax = plt.subplots()
Y.plot(x='time', y='range', cmap='turbo', vmax=20, vmin=0)

X = np.vstack((X.values.ravel(), np.tile(mat_alt, (X.shape[0],1)).ravel()))
X = X.T
Y = Y.values.ravel()
Xtests = X#[np.logical_and(~np.isnan(X[:,0]), ~np.isnan(Y))]
Ytests = Y#[np.logical_and(~np.isnan(X[:,0]), ~np.isnan(Y))]
print(Xtests.shape, Ytests.shape)

# create y predict
Y_pred = tree.predict(Xtests)
Y_pred2 = slope*Xtests[:,0] + intercept
Xtests_features = poly_features.fit_transform(Xtests)
Y_pred3 = reg.predict(Xtests_features)


# In[360]:


fig, ax = plt.subplots()
pcm = ax.pcolormesh(t.values[ii], mat_alt, Y_pred.reshape(alsr355[ii,:].shape).T, 
              cmap='turbo',vmax=20, vmin=0)
plt.colorbar(pcm, ax=ax, extend='both')
plt.xticks(rotation = 45)
title = pd.to_datetime(t.values[ii][0]).strftime('%Y-%m-%d')
ax.set(title = title)
fig.savefig(f'/home/nmpnguyen/ql_predict_trainby{len(Xinput[:,0])}_{title}.png')


# 
# --------------------------------

# ### Vérification les comportements différents entre 2 sub-datasets (par un dataset coupé en deux)

# ---------------------------------------

# In[320]:


alsr355 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['__xarray_dataarray_variable__']
alsr532 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr532-3000-4000.nc')['__xarray_dataarray_variable__']

# Fev2018 to Sep2018
X1 = alsr355[:np.int(alsr355.shape[0]/2),:]#.values
Y1 = alsr532[:np.int(alsr532.shape[0]/2),:]#.values
print(X1.shape)
# Sep2018 to Dec2018
X2 = alsr355[np.int(alsr355.shape[0]/2):,:]#.values 
Y2 = alsr532[np.int(alsr532.shape[0]/2):,:]#.values
print(X2)


# In[317]:


fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(11,4))
h1 = ax.hist2d(X1.values.ravel(), Y1.values.ravel(), bins=100, range=[[-10,40], [-10,80]], norm=LogNorm())
plt.colorbar(h1[3], ax=ax)
ax.set(title=f'from {X1.time.values[0]} \nto {X1.time.values[-1]}',
      xlabel='SR532', ylabel='SR355')
h2 = ax2.hist2d(X2.values.ravel(), Y2.values.ravel(), bins=100, range=[[-10,40], [-10,80]], norm=LogNorm())
plt.colorbar(h2[3], ax=ax2)
ax2.set(title=f'from {X2.time.values[0]} \nto {X2.time.values[-1]}',
       xlabel='SR532', ylabel='SR355')


# In[321]:


X1, Y1 = X1.values, Y1.values
X2, Y2 = X2.values, Y2.values
fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(11,4))
xplot = X1[(X1>5) & (X1<10) & (Y1>-5) & (Y1<5)]
yplot = Y1[(X1>5) & (X1<10) & (Y1>-5) & (Y1<5)]
hh = ax.hist2d(xplot, yplot, bins=100, range=[[-10,40], [-10,80]], norm=LogNorm(vmax=1e2))
plt.colorbar(hh[3], ax=ax)
ax.set(xlabel='SR532', ylabel='SR355')

xplot = X2[(X2>5) & (X2<10) & (Y2>-5) & (Y2<5)]
yplot = Y2[(X2>5) & (X2<10) & (Y2>-5) & (Y2<5)]
hh = ax2.hist2d(xplot, yplot, bins=100, range=[[-10,40], [-10,80]], norm=LogNorm(vmax=1e2))
plt.colorbar(hh[3], ax=ax2)
ax2.set(xlabel='SR532', ylabel='SR355')


# In[244]:


points1 = X1.where((X1>5) & (X1<10) & (Y1>-5) & (Y1<5), drop=False)
points2 = X2.where((X2>5) & (X2<10) & (Y2>-5) & (Y2<5), drop=False)
print(np.count_nonzero(~np.isnan(points1.values), axis=1).shape, points2)


# In[204]:


X2[(X2>10) & (X2<15) & (Y2>-5) & (Y2<5)].shape, Y1[(X1>10) & (X1<15) & (Y1>-5) & (Y1<5)].shape


# In[308]:


points = alsr355.where((alsr355>5) & (alsr355<10) & (alsr532>-5) & (alsr532<5), drop=False)
# points1.resample(time='M').count().sum(axis=1)['time'], points2.resample(time='M').count().sum(axis=1)['time']
print(points.resample(time='M').count().sum(axis=1)['time'], alsr355.resample(time='M').count().sum(axis=1)['time'])
ratio_points = points.resample(time='M').count().sum(axis=1)/alsr355.resample(time='M').count().sum(axis=1)


# In[322]:


print(ratio_points)
get_ipython().magic('matplotlib notebook')
fig, ax = plt.subplots(figsize=(10,6))
# ax.hist(t, bins='auto')
ax.plot(ratio_points['time'].values, ratio_points.values, marker='.')
ax.set(title='(sr355>5) & (sr355<10) & (sr532>-5) & (sr532<5)',
      ylabel='rapport sur nb total/mois')
plt.grid(True)


# --------------------------------------
# ### Test de Decision tree avec 3 features 
# ------------------------------

# In[ ]:


def get_all_data_selected_3features_v2(alldaysfile, nb_profils, z_limite):
    all_X355 = np.array([[], [], []])
    all_X532 = np.array([])

        day_data = xr.open_dataset(day)
#         profils_id = get_profil_random_from_dayfile(day_data, nb_profils)
        X355, Zselected, X532 = get_data_from_profil(day_data, z_limite, nb_profils)
        all_X355 = np.concatenate([all_X355, [X355, Zselected, np.square(Zselected)]], axis=1)
        all_X532 = np.concatenate([all_X532, X532])


# In[397]:


alsr355 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['__xarray_dataarray_variable__'].values
alsr532 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr532-3000-4000.nc')['__xarray_dataarray_variable__'].values
t = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['time'].values

list_random50 = get_profil_random_from_dayfile(t, np.int(alsr532.shape[0]/2))
list_notrandom50 = list(set(range(alsr532.shape[0])) - set(list_random50))


# In[401]:


X = alsr355[list_random50, :]
Y = alsr532[list_random50, :]
mat_alt = np.tile(xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['range'].values, (X.shape[0], 1))
print(mat_alt.shape)
print(X.shape)
print(mat_alt)


# In[392]:


from tqdm import tqdm
X3 = np.zeros(mat_alt.shape)
print(X3.shape)
X3[0,:] = np.nan
for j in tqdm(range(1, mat_alt.shape[1])):
    X3[:,j] = (X[:,j-1] + X[:,j]*(mat_alt[:,j] - mat_alt[:,j-1])).values
print(X3)


# In[399]:


X3.shape


# In[396]:


fig, ax = plt.subplots()
ax.plot(X[206,:], mat_alt[206,:])
ax.plot(X3[206,:], mat_alt[206,:], color='r')


# In[402]:


X = np.vstack((X.ravel(), mat_alt.ravel(), X3.ravel()))
X = X.T
Y = Y.ravel()
print(X.shape)


# In[425]:


Yinput = Y[np.logical_and(~np.isnan(X[:,0]), ~np.isnan(X[:,2]), ~np.isnan(Y))]
Xinput = X[np.logical_and(~np.isnan(X[:,0]), ~np.isnan(X[:,2]), ~np.isnan(Y)),:]

# generate Decision Tree model
from sklearn.tree import DecisionTreeRegressor as DTR
tree = DTR(min_samples_leaf=5)
tree.fit(Xinput,Yinput)


# In[426]:


# generate Data Test 

Xt = alsr355[list_notrandom50, :]
Yt = alsr532[list_notrandom50, :]
mat_alt = np.tile(xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')['range'].values, (Xt.shape[0], 1))
X3t = np.zeros(mat_alt.shape)
print(X3t.shape)
X3t[0,:] = np.nan
for j in tqdm(range(1, mat_alt.shape[1])):
    X3t[:,j] = Xt[:,j-1] + Xt[:,j]*(mat_alt[:,j] - mat_alt[:,j-1])
print(X3t)

Xt = np.vstack((Xt.ravel(), mat_alt.ravel(), X3t.ravel()))
Xt = Xt.T
Yt = Yt.ravel()
Xtests = Xt[np.logical_and(~np.isnan(Xt[:,0]), ~np.isnan(Xt[:,2]), ~np.isnan(Yt)),:]
Ytests = Yt[np.logical_and(~np.isnan(Xt[:,0]), ~np.isnan(Xt[:,2]), ~np.isnan(Yt))]


# In[433]:


Ypred = tree.predict(Xtests)
residus_y_test3features = Ytests - Ypred


# In[434]:


fig, (ax, ax2, ax3) = plt.subplots(ncols=3, figsize=(15,4))
ax.hist2d(Xtests[:,0], Ytests, bins=100, range=[[-10,30], [-10,80]], norm=LogNorm())
ax2.hist2d(Xtests[:,0], Ypred, bins=100, range=[[-10,30], [-10,80]], norm=LogNorm())
ax3.hist2d(Xtests[:,0], residus_y_test3features, bins=100, range=[[-10,30], [-10,80]], norm=LogNorm())


# In[430]:


# évaluation 
residus_y_test3features.describe()


# In[36]:


dtrea=xr.open_dataset('/bdd/ERA5/NETCDF/GLOBAL_025/4xdaily/AN_PL/2013/ta.201312.aphe5.GLOBAL_025.nc')


# In[42]:


print(dtrea)
longitude_selected = [65.5, 65.75, 66, 66.25, 66.5] #je prends au hasard pour l'exemple
ta = dtrea['ta'].sel(longitude =  longitude_selected)
print(ta)


# In[43]:


ta.to_netcdf('/home/nmpnguyen/test_ta.nc')

