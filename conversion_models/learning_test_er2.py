import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

import sys
sys.path.append('/home/nmpnguyen/conversion_model')
import learning_fct_without_sklearn as learning

configs = {
    'base_dir' : Path('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/'),
    'main_dir' : Path('/homedata/nmpnguyen/ORACLES-ER2/RF/Calibrated/'),
    'pattern_filename' : ['HSRL2_ER2_','_R8_v2.nc'],
    'year' : '2016',   
    'variables_name' : {
        'ATB' : 'calibrated', 
        'AMB' : 'molecular', 
        'time' : 'time',
        'height' : 'altitude',
    }, 
    'instrument' : 'ER2',
    'limites' :  [0, 15000], 
        # {'time' : 'flags',
        #  'height' : [4000, 15000], 
        # },
    'wavelenght_from': 532,
    'wavelenght_to' : 355,
    'model_option' : 'create',
}

days_train = ['20160819', '20160823', '20160912', '20160918', '20160922']
days_test = ['20160826', '20160920', '20160924', '20160916']

# before adding features
#-----------------------
traindata = learning.generate_data(days_train, configs, configs['wavelenght_from'])#, mol_traindata
traintarget = learning.generate_data(days_train, configs, configs['wavelenght_to'])#, mol_traintarget


traindata = traindata.sel(time = np.intersect1d(traindata.time.values, traintarget.time.values))
traintarget = traintarget.sel(time = np.intersect1d(traindata.time.values, traintarget.time.values))
# mol_traindata = mol_traindata.sel(time = np.intersect1d(traindata.time.values, traintarget.time.values))
# mol_traintarget = mol_traintarget.sel(time = np.intersect1d(traindata.time.values, traintarget.time.values))

altitude = traindata[configs['variables_name']['height']].values
altitude_2d = np.tile(altitude, (traindata.shape[0], 1))

# adding features
#----------------
from tqdm import tqdm
X3 = np.zeros(traindata.shape)

for j in tqdm(range(altitude.shape[0]-2, 0, -1)):
    delta_z = (altitude_2d[:,j] - altitude_2d[:,j-1])
    X3[:,j] = np.nansum([X3[:,j+1], (traindata[:,j].values*delta_z)], axis=0)   

    
features= {
    0 : altitude_2d.ravel(),
    1 : X3.ravel()
}

new_traindata = learning.generate_feature(traindata.values.ravel(), features)

features={}
new_traintarget = learning.generate_feature(traintarget.values.ravel(), features)

# add features for molecular data and clean data
#-------
features={}
# new_moltrain = learning.generate_feature(mol_traindata.values.ravel(), features)
# new_moltarget = learning.generate_feature(mol_traintarget.values.ravel(), features)
clean_traindata, clean_traintarget, idmask = learning.clean_data_target(new_traindata, new_traintarget)

# saving
#-------
pd_train_all = pd.DataFrame(np.concatenate([new_traindata, new_traintarget], axis=1), columns=['sr355', 'alt', 'sr355_integrated', 'sr532']) #, new_moltrain, new_moltarget, 'mol355', 'mol532'

output_filename = 'pd_train_all.pkl'
pd_train_all.to_pickle(Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/',output_filename))


# pd_train_all = pd.read_pickle(Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/','pd_train_all_ATB-to-SR.pkl'))
# clean_traindata, clean_traintarget, idmask = learning.clean_data_target(pd_train_all[['data355', 'alt', 'data355_integrated']].to_numpy(), pd_train_all[['data532']].to_numpy())


print('-----------------------MODEL--------------------------')
if configs['model_option'] == 'read':
    import pickle
    with open('/home/nmpnguyen/conversion_model/tree_3f.sav', 'rb') as handle:
        model_loaded = pickle.load(handle)
else:
    from sklearn.tree import DecisionTreeRegressor
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.pipeline import make_pipeline
    # from sklearn.model_selection import GridSearchCV
    '''
    training without preprocessing
    -------------------------------
    '''
    model_loaded = DecisionTreeRegressor()  
    '''
    training with preprocessing
    -------------------------------
    '''
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.pipeline import make_pipeline
    # model_loaded = make_pipeline(StandardScaler(), DecisionTreeRegressor())
    '''
    traing with grid search
    '''
    # model_loaded = RandomForestRegressor(n_estimators = 100, max_depth = 7)
    # model_loaded.fit(clean_traindata, clean_traintarget.ravel())
    # from sklearn.model_selection import cross_val_score
    # scores_random_forest = cross_val_score(model_loaded, clean_traindata, clean_traintarget.ravel())
    # print(f"Random forest regressor: "
    #   f"{scores_random_forest.mean():.3f} ± "
    #   f"{scores_random_forest.std():.3f}")
    
#     param_grid = {
#         'max_depth' : (3, 7, 15, 30),
#         'min_samples_leaf' : (5, 10, 15, 20, 30),
#     }

#     model_grid_search = GridSearchCV(
#         model_loaded, param_grid=param_grid, n_jobs=-1, cv=10
#     )

#     model_grid_search.fit(clean_traindata, clean_traintarget)
#     print(model_grid_search.best_params_)
model_loaded.fit(clean_traindata, clean_traintarget)

mae_test = []
r2_test = []
pd_test_all = None

from sklearn.metrics import mean_absolute_error, r2_score
for day in days_test:
#     try : 
    test355 = learning.generate_data([day], configs, 355)#, mol_test355
    test532 = learning.generate_data([day], configs, 532)#, mol_test532
    test355 = test355.sel(time = np.intersect1d(test355.time.values, test532.time.values))
    test532 = test532.sel(time = np.intersect1d(test355.time.values, test532.time.values))
    # test355 = test355.where(test355 > 5, drop=False)
    # test532 = test532.where(test532 > 5, drop=False)
    # mol_test355 = mol_test355.sel(time = np.intersect1d(test355.time.values, test532.time.values))
    # mol_test532 = mol_test532.sel(time = np.intersect1d(test355.time.values, test532.time.values))
    
    altitude = test355.altitude.values
    altitude_2d = np.tile(altitude, (test355.shape[0], 1))

    #generate features for X
    #-----------------------
    from tqdm import tqdm
    X3t = np.zeros(test355.shape)
    for j in tqdm(range(altitude.shape[0]-2, 0, -1)):
        delta_z = (altitude_2d[:,j] - altitude_2d[:,j-1])
        X3t[:,j] = np.nansum([X3t[:,j+1], (test355[:,j].values*delta_z)], axis=0)
        
    features= {
        0 : altitude_2d.ravel(),
        1 : X3t.ravel()
    }
    new_test355 = learning.generate_feature(test355.values.ravel(), features)
    #generate features for Y
    #------------------------
    features={}
    new_test532 = learning.generate_feature(test532.values.ravel(), features)
    #------------------------
    clean_test355, clean_test532, idmask = learning.clean_data_target(new_test355, new_test532)
    #------------------------
    # new_mol_test355 = learning.generate_feature(mol_test355.values.ravel(), features)
    # new_mol_test532 = learning.generate_feature(mol_test532.values.ravel(), features)
    # print('clean mol')    
    # clean_mol_test355, clean_mol_test532, _ = clean_data_target(new_mol_test355, new_mol_test532)
    #------------------------
    predict_test532 = model_loaded.predict(clean_test355)
    #
    print(f'MAE : {mean_absolute_error(clean_test532, predict_test532)}')
    mae_test.append(mean_absolute_error(clean_test532, predict_test532))
    r2_test.append(r2_score(clean_test532, predict_test532))
    #
    newpredict_test532 = np.full(test532.sel(time = test355.time).values.ravel().shape, np.nan)
    newpredict_test532[np.where(idmask)] = predict_test532
    newpredict_test532 = newpredict_test532.reshape(test532.sel(time = test355.time).shape)
    #
    newpredict_test532 = xr.DataArray(newpredict_test532, 
                              coords = test532.coords)
#     ds = xr.merge([xr.DataArray(test355, name='SR355_measured', coords = test532.coords), 
#                    xr.DataArray(X3t, name='SR355_integrated', coords = test532.coords),
#                    xr.DataArray(test532, name='SR532_measured', coords = test532.coords),
#                    xr.DataArray(newpredict_test532, name = 'SR532_predicted', coords = test532.coords)],
#                   compat='override')
#     ds.to_netcdf(f'/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/tree_3f-HSRL2-ER2-{day}.nc')
#         cmap = plt.cm.turbo
#         cmap.set_under('lightgrey')
#         cmap.set_over('dimgrey')
#     except:
#         pass
    # new_test355[:,0]=new_test355[:,0]/new_mol_test355.flatten()
    # new_test532 = new_test532/new_mol_test355

    pd_test = pd.DataFrame(np.concatenate([new_test355, new_test532, (newpredict_test532.values.flatten()).reshape(-1,1)], axis=1), columns=['sr355', 'alt', 'sr355_integrated', 'sr532', 'sr532_predicted'])
    # , new_mol_test355, new_mol_test532 , 'mol355', 'mol532'
    pd_test['absolute_error'] = (pd_test['sr532']-pd_test['sr532_predicted'])
    if pd_test_all is None:
        pd_test_all = pd_test   
    else:        
        pd_test_all = pd.concat([pd_test_all, pd_test])
    
output_filename = 'pd_test_all.pkl'
pd_test_all.to_pickle(Path('/homedata/nmpnguyen/ORACLES-ER2/leaning_model_test/', output_filename))