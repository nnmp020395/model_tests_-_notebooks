import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

import sys
sys.path.append('/homedata/nmpnguyen/database_lidars/Codes')
from outils import convert_gpstime

def train_test_days_split(list_days, test_size, random=False):
    counts_days = len(list_days)
    counts_train_days = np.int((1-test_size)*counts_days)
    counts_test_days = counts_days - counts_train_days
    if random:
#         ind_train_days = np.random.randint(0,counts_days,counts_train_days)
        ind_train_days = np.random.choice(np.arange(counts_days), counts_train_days, replace=False)
        train_days = np.array(list_days)[ind_train_days]
        ind_test_days = np.delete(np.arange(counts_days), ind_train_days)
        test_days = np.array(list_days)[ind_test_days]
    else:
        train_days = list_days[:counts_train_days]
        test_days = list_days[counts_train_days:]
    print('Lenght of train days:', len(train_days))
    print('Lenght of test days:', len(test_days))
    return train_days, test_days



class er2_class:
    coords_name = {'time':'time', 'height' : 'altitude'}
    def __init__(self):
        self.time = 'time'
        self.height = 'altitude'
    
    def get_data(self, path, wavelength, limites):
        file = xr.open_dataset(path)
        convert_timelist = convert_gpstime(file[self.time].values, path.stem.split('_')[2], convert=True)
        file_single = file.assign_coords(time = convert_timelist)
        file_single = file_single.dropna(self.time, how='all')
        limites_height = np.where(np.logical_and(file_single[self.height].values > limites[0],file_single[self.height].values < limites[1]))[0]
        file_single = file_single.sel(wavelength=wavelength).isel(altitude = limites_height)
        return file_single

    
    
class ipral_class:
    def __init__(self, path, wavelength, selected_height):
        self.path = path
        self.time = 'time'
        self.height = 'range'
        self.flags = 'flags'
        self.selected_height = selected_height
        self.wavelength = wavelength

    def generate_data_with_limites(self):
        file = xr.open_dataset(self.path)
        #---------------
        flags = file.where(file[self.flags].sel(wavelength=self.wavelength) == 0, drop=True)[self.time].values
        limites_height = np.where((file[self.height].values > self.selected_height[0]) & (file[self.height].values < self.selected_height[1]))[0]
        file_single = file.sel(wavelength=self.wavelength, time = flags).isel(range = limites_height)
        file_single = file_single.resample(time = '15min', skipna=True).mean(self.time)
        return file_single
    
    

def generate_data(list_days, configs, wavelength_from, wavelength_to):
    '''
    from list day, go to the directory of filted calibrated data
    '''    
    # get all paths from list_days
    #------------------------------
    def get_all_path(list_days, configs):
        list_paths = []
        for day in list_days:
            filename = configs['pattern_filename'][0]+pd.to_datetime(day).strftime('%Y%m%d')+configs['pattern_filename'][1]
            try:
                list_paths.append(sorted(Path(configs['main_dir']).rglob(filename))[0])
            except:
                pass
        return list_paths    

    def get_all_data(list_paths, instrument, variable_name, wavelength, limites, concat_dim):
    # get all sr from list_days
    #--------------------------        
        allData = []
        for path in list_paths:
            try:
                # file = xr.open_dataset(path)
                # if instrument is ER2
                #---------------------            
                if instrument == "ER2":
                    print('instrument=ER2')
                    # for_er2 = er2_class()
                    # convert_timelist = for_er2.convert_gpstime(file.time.values, path.stem.split('_')[2], convert=True)
                    # file = file.assign_coords(time = convert_timelist)
                    tool = er2_class()
                    file = tool.get_data(path, wavelength, limites)
                elif instrument == 'IPRAL':
                    print('instrument=IPRAL')
                    tool = ipral_class(path = path, wavelength = wavelength, selected_height = limites)
                    file = tool.generate_data_with_limites()
                   
                data = file[variable_name]
                #---------------------  

                # if wavelength is None:
                #     if variable_name == 'flags':
                #         data = file['time'].where(file['flags'] == 0, drop=True)
                #     else:
                #         data = file[variable_name]#.dropna(dim='time', how='all')                    
                # else:
                #     if variable_name == 'flags':
                #         data = file['time'].where(file['flags'].sel(wavelength=wavelength) == 0, drop=True)
                #     else:
                #         data = file[variable_name].sel(wavelength=wavelength)#.dropna(dim='time', how='all')
                
                #---------------------    
                data = data.dropna(dim='time', how='all')
                allData.append(data)
            except (FileNotFoundError, IndexError):
                print('Cannot found filepath')
                pass
            
        if len(allData) == 0:
            print('Cannot concatenate')
            return 1
        else:
            allData = xr.concat(allData, dim=concat_dim)   
            return allData
    
    list_paths = get_all_path(list_days, configs)
    print(f'List of files: {list_paths}')

    time_name = configs['variables_name']['time']
    # try : 
    ATB_from = get_all_data(list_paths, configs['instrument'], configs['variables_name']['ATB'], wavelength_from, configs['limites'], concat_dim=time_name)
    AMB_from = get_all_data(list_paths, configs['instrument'], configs['variables_name']['AMB'], wavelength_from, configs['limites'], concat_dim=time_name)
    SR_from = ATB_from/AMB_from

    ATB_to = get_all_data(list_paths, configs['instrument'], configs['variables_name']['ATB'], wavelength_to, configs['limites'], concat_dim=time_name)
    AMB_to = get_all_data(list_paths, configs['instrument'], configs['variables_name']['AMB'], wavelength_to, configs['limites'], concat_dim=time_name)
    SR_to = ATB_to/AMB_to
    # if np.isin('flags', list(configs['variables_name'].keys())):
    #     flags = get_all_data(list_paths, configs['instrument'], configs['variables_name']['flags'], wavelength, concat_dim=time_name)
    #     ATB = ATB.sel(time = flags)
    #     AMB = AMB.sel(time = flags)
    
    return SR_from, SR_to
    # except:
    #     pass


def generate_feature(data, features_data): 
    # input all data needed to generate on feature       
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
        
    nb_features = len(features_data)
    if nb_features == 0:
        print('Any feature to adding')
        data = data.reshape(-1,1)
    else:
        for i in range(len(features_data)):
            print(f'add feature numero {i}')
            data = add_feature(data, features_data[i])

    return data


def clean_data_target(data, target):
    '''
    Use to clean Nan/Inf or negative values of data
    '''
    print(f'Before: data = {data.shape}, target = {target.shape}')
    # data 
#     data = pd.DataFrame(data)
    mask_data = np.isfinite(data).all(1)
    # target
#     target = pd.DataFrame(target)
    mask_target = np.isfinite(target).all(1)
    # intersection
    mask = np.logical_and(mask_data, mask_target)
    print(mask)
#     mask = np.logical_and(np.isfinite(X).all(1), np.isfinite(Y.values.ravel()))
    print(f'shape of mask array {mask.shape}')
    new_data = data[mask, :]
    new_target = target[mask]
    print(f'After : new_data = {new_data.shape}, new_target = {new_target.shape}')
    return new_data, new_target, mask 


# def cross_validation():
#     return 1

# def test_score(model, target, predict):
#     '''
#     return mean_absolute_error and r2_score between data target and data predict
#     '''
#     from sklearn.metrics import r2_score, mean_absolute_error 
#     print('Check data before compute')
    
#     return 1
