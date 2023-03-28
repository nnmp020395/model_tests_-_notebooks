# Basics librairies
#------------------
import numpy as np
import pandas as pd 
import xarray as xr 
from pathlib import Path 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime

# Libraires for Decision Tree model 
#----------------------------------

from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def add_feature(data_before, adding):
    '''
    function is used to add features create test/train/validation dataset
    '''
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

def get_prediction(dtX_input, model_input):
    '''
    function is used to predict X input data
    '''
    # select only no-Nan values
    #--------------------------
    indx = np.logical_and(~np.isnan(dtX_input[:,0]), ~np.isnan(dtX_input[:,2]))
    print(f'Index : {indx}')
    dtX_input2 = dtX_input[indx,:] #
    # indx2 = np.where(np.isnan(dtX_input[np.unique(indx[0]),:]))
    # print(f'Intersection :{np.intersect1d(indx[0], indx2[0])}')
    # predict
    #--------
    dt_predict = model_input.predict(dtX_input2)
    dt_predict_new = np.full(dtX_input.shape[0], np.nan)
    dt_predict_new[indx] = dt_predict
    print(f'Check shape of predicted data and input test : {dt_predict_new.shape, dtX_input.shape}')
    return dt_predict_new

def performance(model_input, Xinput_val, Yinput_val, Ypred_val):
    residus_val = Yinput_val - Ypred_val
    stats_residus_val = pd.DataFrame(residus_val).describe()#[['mean', 'std']]
    return stats_residus_val #mean_residus, std_residus



# Input calibrated data
#----------------------

from argparse import Namespace, ArgumentParser
parser = ArgumentParser()
parser.add_argument('--filepath', '-file', required=True, type=str, help='Path of calibrated data for prediction')
parser.add_argument('--modelname', '-model', required=True, type=str, help='Name of loaded model')
opts = parser.parse_args()

# Load Model
#------------------------------

import pickle
loaded_model = pickle.load(open(f'/home/nmpnguyen/conversion_model/{opts.modelname}.sav', 'rb'))

# Prepare dataset for test data 
#------------------------------

file_path = Path(opts.filepath)
print(file_path)
# Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/zone-3000-4000/ipral_1a_Lz1R15mF30sPbck_v01_20200904_000000_1440.nc')
dt = xr.open_dataset(file_path)

dtalt = dt['range'].values
dttime = dt['time'].values
convert_choice = '0'

if convert_choice == '0':
### 3 FEATURES 
    dtsr = (dt['calibrated']/dt['simulated']).sel(wavelength=355).where(dt['flags'].sel(wavelength=355) == 0, drop=False).values
    dtX2 = add_feature(dtsr.ravel(), np.tile(dtalt, dtsr.shape[0])) 
    dtalt_mat = np.tile(dtalt, (dtsr.shape[0],1))
    # 3e feature of X_mat
    from tqdm import tqdm
    X3 = np.zeros(dtalt_mat.shape)
    print(X3.shape)
    # X3[0,:] = np.nan
    for j in tqdm(range(1, dtalt_mat.shape[1])):
        X3[:,j] = X3[:,j-1] + dtsr[:,j]*(dtalt_mat[:,j] - dtalt_mat[:,j-1])

    
    dtX = add_feature(dtX2, X3.ravel())

    dtY = (dt['calibrated']/dt['simulated']).sel(wavelength=532).where(dt['flags'].sel(wavelength=532) == 0, drop=False).values
    plot_title = '355_to_532'

else:
    print('convert 355 from 532 signal')
    # dtsr = (dt['calibrated']/dt['simulated']).sel(wavelength=532).values
    dtsr = (dt['calibrated']).sel(wavelength=532).values
    dtX = add_feature(dtsr.ravel(), np.tile(dtalt, dtsr.shape[0]))
    # dtY = (dt['calibrated']/dt['simulated']).sel(wavelength=355).values
    dtY = (dt['calibrated']).sel(wavelength=355).values
    plot_title = '532_to_355'


dtPredict = get_prediction(dtX, loaded_model)
dtPredict = dtPredict.reshape(dtsr.shape)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
scores = mean_absolute_error(dtY, dtPredict)
print(f'mean_absolute_error : {scores}')

coords = {
    'time' : (['time'], dttime),
    'range' : (['range'], dtalt)
    }
attrs = {
    'filename' : 'tmp file of prediction dataset',
    'creation_datetime' : str(datetime.now()),
    'mean_absolute_error' : mean_absolute_error(dtY, dtPredict),
    'mean_squared_error' : mean_squared_error(dtY, dtPredict),
    'r2_score' : r2_score(dtY, dtPredict)
    }
data_vars = {
    'X' : (['time', 'range'], dtsr),
    'Y' : (['time', 'range'], dtY),
    'Ypredicted' : (['time', 'range'], dtPredict)
}
dsPredict = xr.Dataset(data_vars=data_vars, 
                        coords=coords, 
                        attrs=attrs)

dsPredict.to_netcdf(f'{file_path.stem}_conversion_test_{plot_title}.nc')




