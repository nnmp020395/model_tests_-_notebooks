
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# In[64]:


'''
Fct de conversion théorique 
'''

def conversion_theorical(input_sr355, coef=5.3):
    print('Theorical coef:', coef)
    print('Shape of sr355 input:', input_sr355.shape)
    output_array532 = coef * input_sr355
    print('Shape of sr532 output', output_array532.shape)
    return output_array532


# In[65]:


class conditions:
    def __init__(self, value: float, close: str, id_where):
        self.value = value
        self.close = close 
        self.where = id_where
#         return self
    


# In[66]:


import numpy.ma as ma  
class dataset:
    def __init__(self, calibrated_variable: str, simulated_variable: str, 
                 flag_variable: str, flag_value_valid: int, limite_altitude: float):
        self.calibrated_variable = calibrated_variable
        self.simulated_variable = simulated_variable
        self.flag_variable = flag_variable
        self.flag_value_valid = flag_value_valid
        self.limite_altitude = limite_altitude

#------------------------------------------

'''
Fct de récupération des données (1 profil, 1 dataset) et 
sélection en fonction de condition (input by user)
'''
class get_file:
    def get_closed_data(input_data, value, close):
        if close == 'left':                
            output_data = ma.masked_where((input_data > 0) & (input_data < value), input_data)
            output_data = output_data.filled(fill_value=np.nan)
            id_where = np.where(input_data > value)
            print(id_where)
        elif close == 'right':
            output_data = ma.masked_where(input_data > value, input_data)
            output_data = output_data.filled(fill_value=np.nan)   
            id_where = np.where(input_data < value)
            print(id_where)
        elif close == 'both':
            print()
            id_where = []
        elif (close is None):
            output_data = input_data
            id_where = []
        return output_data, id_where


    def set_dataset_with_condition(input_data, condition=None, return_index=False):   

        print('Shape of input:', input_data.shape)    
        print('Conditions for apply: ', condition.value, condition.close, condition.where)
        #------------------------
        if (condition.where is not None):
            print('1')
            id_where = condition.where
            data_where = input_data[id_where]
            output_data, _ = get_file.get_closed_data(data_where, condition.value, condition.close)            
        else: 
            print('2')
            output_data, id_where = get_file.get_closed_data(input_data, condition.value, condition.close)
             
        #------------------------    
        print('Shape of output after apply conditions:', output_data.shape)
        if (return_index):
            return output_data, id_where
        else:
            return output_data


    def get_file_dataset(mainfile, characters, wavelength, conditionf=None, return_index = False):
        '''
        dataset should be the same format: 
            - netCDF
            - variable: calibrated, simulated, flags
        '''
        input_data = xr.open_dataset(mainfile)
        limites_range = input_data['range'][input_data['range'] < characters.limite_altitude]
        sr_data = input_data[characters.calibrated_variable].sel(wavelength=wavelength, range=limites_range)/input_data[characters.simulated_variable].sel(wavelength=wavelength, range=limites_range)
        flagged_data = sr_data.where(input_data[characters.flag_variable].sel(wavelength=wavelength)==characters.flag_value_valid, drop=False)
        output_data = flagged_data.resample(time = '15min', skipna=True).mean('time')
        # print('get_file_dataset', conditionf.where)
        final_output_data = get_file.set_dataset_with_condition(output_data, conditionf, return_index = False)          
        return final_output_data



    def get_folder_dataset(mainfolder: str, patternfile: str, characters, wavelength, 
                           grouped=False, conditionF=None, return_index=False):
        from tqdm import tqdm
        listfiles = sorted(Path(mainfolder).glob(patternfile))
        outputs_data = []
        if grouped:
            for file in tqdm(listfiles):
                print(file)          
                output_1_data = get_file.get_file_dataset(file, characters, wavelength, conditions(np.nan, None, None), False)
                outputs_data.append(output_1_data)
        
            grouped_outputs_data = np.concatenate(outputs_data, axis=0)

            # check y_shape of grouped output data
            #-----------------------------------    
            if (output_1_data.shape[1] == grouped_outputs_data.shape[1]):
                print('Shape of output data after groupping', grouped_outputs_data.shape)
                print('------------Groupping: Done-------------')
                # return grouped_outputs_data
            else:
                print('------------Groupping: Error-------------')
                print('Shape of 1 output:', output_1_data.shape)
                print('Shape of list outputs:', outputs_data.shape)
                return 0
        else:
            for file in tqdm(listfiles):
                print(file)    
                print('before', conditionF.where)        
                output_1_data = get_file.get_file_dataset(file, characters, wavelength, None, True)
                print('after', conditionF.where)
                outputs_data.append(output_1_data)

            grouped_outputs_data = outputs_data
            print('Shape of output data without groupping', grouped_outputs_data.shape)
            print('------------Groupping: Done-------------')

        final_output_data, ids_where = get_file.set_dataset_with_condition(grouped_outputs_data, conditionF, return_index=True)
        print('Shape of output data after setting conditions', final_output_data.shape)
        print('------------Setting: Done-------------')
        return final_output_data, ids_where


# In[67]:


ipral_characters = dataset('calibrated', 'simulated', 'flags', 0, 20000)

#---------------------
dataset_name = 'IPRAL2020'
maindir = '/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/2020/'
pattern = 'ipral_1a_Lz1R15mF30sPbck_v01_2020*_000000_1440.nc'
#---------------------

# THEORICAL METHOD
method_name = 'THEORICAL'

print(f'----------------GENERATE DATA--------------------')
condt = conditions('None', 'None', None)
print('----355')


# In[6]:


dataset355, ids = get_file.get_folder_dataset(maindir, pattern, ipral_characters, 355, grouped=True, conditionF=condt, return_index=True)
# print(ids)


# In[8]:


# saving
import pickle
home_dir = '/homedata/nmpnguyen/comparaison/'
# dataset355 = dataset355[ids]
with open(Path(home_dir,f'{dataset_name}-355-mes_{method_name}_CONDT-SR355-{condt.value}-{condt.close}.pkl'), 'wb') as output_dataset:
    pickle.dump(dataset355, output_dataset)  


# In[9]:


dataset355#.shape, dataset355[ids]


# In[10]:


condt2 = conditions(1.0, 'right', ids)
print('----532')
dataset532mes = get_file.get_folder_dataset(maindir, pattern, ipral_characters, 532, True, condt2, False)


# In[12]:


dataset532mes[0].shape, dataset355.shape


# In[61]:


# saving
with open(Path(home_dir,f'{dataset_name}-532-mes_{method_name}_CONDT-SR355-{condt.value}-{condt.close}.pkl'), 'wb') as output_dataset:
    pickle.dump(dataset532mes[0], output_dataset)


# In[13]:


dataset532pred = conversion_theorical(dataset355, 5.3)
# saving
# with open(Path(home_dir,f'{dataset_name}-532-pred_{method_name}_CONDT-SR355-{condt.value}-{condt.close}.pkl'), 'wb') as output_dataset:
#     pickle.dump(dataset532pred, output_dataset)


# In[16]:


NotNans = np.logical_and(~np.isnan(dataset532mes[0].ravel()), ~np.isnan(dataset532pred.ravel()))
# dataset532mes[0].ravel()
# dataset532mes = dataset532mes[0].ravel()[NotNans]
# dataset532pred = dataset532pred.ravel()[NotNans]


# In[28]:


negative_ids = np.logical_and(dataset532mes[0].ravel()[NotNans]>0, dataset532pred.ravel()[NotNans]>0)

new_dataset532mes = dataset532mes[0].ravel()[NotNans][negative_ids]
new_dataset532pred = dataset532pred.ravel()[NotNans][negative_ids]


# In[36]:


plt.hist2d(new_dataset532mes, new_dataset532pred, bins=100, norm=LogNorm())


# In[38]:


class plots:
    '''
    Script sert à créer l'histogramme avec le colorbar représentant 
    la proprortion de la distribution des points et la ligne diagonale du plot 
    '''
    def __init__(self, method_name, dataset_name, closed_units, min_value, max_value, mesures, predictes, 
        captions, labels, output_path):
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.closed_units = closed_units
        bins_array = np.arange(min_value, max_value, 0.1)
        H, self.xedges, self.yedges = np.histogram2d(mesures, predictes, bins = [bins_array, bins_array])
        self.bins_array = bins_array
        Hpercents = (H/mesures.shape[0])*100
        self.Hpercents = Hpercents
        self.labels = labels
        self.captions = captions
        self.output_path = output_path
        self.min_value = min_value
        self.max_value = max_value

    def precent_hist2d(self):
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        p = ax.pcolormesh(self.xedges, self.yedges, (self.Hpercents).T, norm=LogNorm(vmin=1e-3, vmax=1e0))
        cbar = plt.colorbar(p, ax=ax, extend='both', label='Probability, %')
        cbar.ax.tick_params(labelsize=13)
        # the diagonal line
        ax.plot(self.bins_array, self.bins_array, '-r')
        ax.plot(self.bins_array + self.closed_units[-1], self.bins_array, linestyle ='--', label = f'+/- {self.closed_units[-1]}', color = 'red')
        ax.plot(self.bins_array - self.closed_units[-1], self.bins_array, linestyle ='--', color = 'red')
        ax.legend()
        # grid
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
        ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
        # title
        plt.suptitle(f'{self.method_name}-{self.dataset_name}', ha='left', fontsize=16)
        # subtitle        
#         title_chains = [f'+/- {unit} : {np.round(sts, decimals=2)}% \n' for unit,sts in zip(self.closed_units, self.stats)]
#         title_chains.append(self.captions[0])
        title_chains = [self.captions]
        plt.title(" ".join(title_chains), loc='left', fontsize=11)
        # AXIS LABELS
        plt.ylabel(self.labels[1], fontsize=13)
        plt.xlabel(self.labels[0], fontsize=13)
        ax.tick_params(labelcolor='k', labelsize=13)
        # CAPTION
#         plt.text(-0.5, -10.5, self.captions[0], ha='left', fontsize = 11, alpha=0.9)

        # X-LIM & Y-LIM
        plt.xlim(self.min_value, self.max_value)
        plt.ylim(self.min_value, self.max_value)
        plt.tight_layout()
#         print(Path(self.output_path, f'{self.dataset_name}_{self.method_name}_{self.captions[1]}.png'))
#         plt.savefig(Path(self.output_path, f'{self.dataset_name}_{self.method_name}_{self.captions[1]}.png'))
#         plt.close()


# In[12]:


dataset_name = 'IPRAL2020'
years = ['2018', '2019', '2020']
maindir = '/homedata/nmpnguyen/IPRAL/NETCDF/v_simple'
pattern = 'ipral_1a_Lz1R15mF30sPbck_v01_*_000000_1440.nc'
# all_sr355 = []
# all_sr532 = []
for year in years:
    for file in sorted(Path(maindir, year).glob(pattern)):
        print(file)
#         dt = xr.open_dataset(file)
#         z = (dt['range'] < 20000)

#         sr355 = (dt['calibrated']/dt['simulated']).isel(range = z, wavelength=0)
#         sr355 = sr355.where(dt['flags'].isel(wavelength=0) == 0, drop=True)   
#         sr532 = (dt['calibrated']/dt['simulated']).isel(range = z, wavelength=1)
#         sr532 = sr532.where(dt['flags'].isel(wavelength=1) == 0, drop=True)
#         if (sr355.shape[0] != 0) | (sr532.shape[0] != 0):
#             sr355 = sr355.resample(time = '15min').mean('time', skipna=True)
#             sr532 = sr532.resample(time = '15min').mean('time', skipna=True)

#         all_sr355.append(sr355)
#         all_sr532.append(sr532)


# In[155]:


all_sr355 = xr.concat(all_sr355, dim='time')
all_sr532 = xr.concat(all_sr532, dim='time')


# In[156]:


a0, a = np.intersect1d(all_sr355['time'], all_sr532['time'], return_indices=True)[1:]
all_sr355 = all_sr355.isel(time=a0)
all_sr532 = all_sr532.isel(time=a)


# In[24]:


# pattern = ['/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr',
#           '3000-4000.nc']
pattern = ['/homedata/nmpnguyen/comparaison/IPRAL2018_IPRAL2019_IPRAL2020-', 
           'mes_THEORICAL_CONDT-None-None-None.nc']
dataset_name = 'IPRAL2018'
all_sr355 = xr.open_dataarray(f'{pattern[0]}355-{pattern[1]}')
all_sr532 = xr.open_dataarray(f'{pattern[0]}532-{pattern[1]}')


# In[20]:


all_sr532.shape, all_sr355.shape


# In[21]:


plt.hist2d(all_sr532.values.ravel(), all_sr355.values.ravel(), norm=LogNorm(vmax=1e5), range=[[0,80], [0,80]], bins=100, density=False)
plt.colorbar()
# plt.title('all_sr355.values.ravel(), all_sr532.values.ravel()')
plt.title('THEORITICAL-IPRAL-2018/2019/2020')
plt.ylabel('355 mesures')
plt.xlabel('532 mesures')

# params_plot = plots('THEORICAL', dataset_name, np.arange(0.01,1,0.05), 0, 50, all_sr355.values.ravel(), all_sr532.values.ravel(),
#                     f'{(~np.isnan(all_sr532.values[all_sr355.values>0])).sum()} points total', ['355 mesures', '532 mesures'], ' ')
# params_plot.precent_hist2d()


# In[189]:


all_sr355.values.ravel()*5.3


# In[62]:


for const in [4.0, 5.3]: 
    fig, ax = plt.subplots()
    p = ax.hist2d(all_sr532.values.ravel(), all_sr355.values.ravel()*const, norm=LogNorm(vmax=1e5), range=[[0,80], [0,80]], bins=100, density=False)
    plt.colorbar(p[3], ax=ax)
#     ax.plot(np.arange(0,80,1), np.arange(0,80,1), color='r')
    ax.plot(np.arange(0,80,1)+2, np.arange(0,80,1), color='r')
    ax.plot(np.arange(0,80,1)-2, np.arange(0,80,1), color='r')
    # plt.title('all_sr355.values.ravel(), all_sr532.values.ravel()')
    ax.set_title(f'THEORITICAL {const}-IPRAL-2018/2019/2020-')
    ax.set_ylabel('532 predict')
    ax.set_xlabel('532 mesures')


# In[70]:


#---------------SR355 

#__init__(self, method_name, dataset_name, closed_units, min_value, max_value, mesures, predictes, 
#         captions, labels, output_path)
params_plot = plots('THEORICAL', dataset_name, np.arange(0.01,1,0.05), 0, 50, all_sr532.values.ravel(), all_sr355.values.ravel()*5.3,
                    f'{(~np.isnan(all_sr532.values[all_sr355.values>0])).sum()} validated points total', ['532 mesures', '532 predict'], ' ')
params_plot.precent_hist2d()

# plt.hist2d(all_sr532.values.ravel(), all_sr355.values.ravel()*5.3, norm=LogNorm(), range=[[0,80], [0,80]], bins=100, density=True)
# plt.colorbar()
# plt.title('all_sr532.values.ravel(), all_sr355.values.ravel()*5.3')


# In[71]:


#---------------SR355 > 1
# plt.hist2d(all_sr532.values[all_sr355.values>1], all_sr355.values[all_sr355.values>1]*5.3, norm=LogNorm(), range=[[0,80], [0,80]], bins=100, density=True)
# plt.colorbar()
params_plot = plots('THEORICAL', dataset_name, np.arange(0.01,1,0.05), 1, 50, all_sr532.values[all_sr355.values>1], all_sr355.values[all_sr355.values>1]*5.3,
                   f'{(~np.isnan(all_sr532.values[all_sr355.values>1])).sum()} validated points total', ['532 mesures', '532 predict'], ' ')
params_plot.precent_hist2d()


# In[22]:


all_sr355.to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2020-355-mes_THEORICAL_CONDT-None-None-None.nc')
all_sr532.to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2020-532-mes_THEORICAL_CONDT-None-None-None.nc')

all_sr532.where(all_sr355 > 1.0, drop=True).to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2020-532-mes_THEORICAL_CONDT-SR355-1.0-left.nc')
all_sr532.where(all_sr355 < 1.0, drop=True).to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2020-532-mes_THEORICAL_CONDT-SR355-1.0-right.nc')

all_sr355.where(all_sr355 > 1.0, drop=True).to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2020-355-mes_THEORICAL_CONDT-SR355-1.0-left.nc')
all_sr355.where(all_sr355 < 1.0, drop=True).to_netcdf('/homedata/nmpnguyen/comparaison/IPRAL2020-355-mes_THEORICAL_CONDT-SR355-1.0-right.nc')


# In[72]:


#---------------SR355 < 1
# plt.hist2d(all_sr532.where(all_sr355 < 1.0, drop=True).values.ravel(), all_sr355.where(all_sr355 < 1.0, drop=True).values.ravel()*5.3, norm=LogNorm(), range=[[0,80], [0,80]], bins=100, density=True)
# plt.colorbar()

params_plot = plots('THEORICAL', 'IPRAL2020', np.arange(0.01,1,0.05), 0, 10, all_sr532.values[(all_sr355.values>0)&(all_sr355.values<1)], all_sr355.values[(all_sr355.values>0)&(all_sr355.values<1)]*5.3,
                   f'{(~np.isnan(all_sr532.values[(all_sr355.values>0)&(all_sr355.values<1)])).sum()} validated points total', ['532 mesures', '532 predict'], ' ')
params_plot.precent_hist2d()


# ### LEARNING METHOD

# In[77]:


TEST_dataset = pd.read_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/','ipral_2018-2019-2020_learned_train_dataset.pkl'))
TESTtarget_dataset = pd.read_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/','ipral_2018-2019-2020_learned_traintarget_dataset.pkl'))


# In[79]:


plt.hist2d(TEST_dataset.values[:,0], TESTtarget_dataset.values.ravel(), norm=LogNorm(vmax=1e5), range=[[0,80], [0,80]], bins=100, density=False)
plt.colorbar()
plt.title('testtarget_dataset.values.ravel(), predict_dataset.values.ravel()')
plt.xlabel('355 mesures')
plt.ylabel('532 mesures')
plt.title('LEARNING-IPRAL-2018-2019-2020')


# In[71]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(TEST_dataset, TESTtarget_dataset)


# In[73]:


import pickle

with open('/home/nmpnguyen/conversion_model/comparaison/comparaison_model.sav', 'wb') as f:
    pickle.dump(model, f)


# In[84]:


test_dataset = pd.read_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/','ipral_2018-2019-2020_learned_train_dataset.pkl'))
testtarget_dataset = pd.read_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/','ipral_2018-2019-2020_learned_traintarget_dataset.pkl'))

# -----------------test_dataset -> prediction where all sr355----------------- 
# predict_dataset = model.predict(test_dataset)
# pd.DataFrame(predict_dataset).to_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/','ipral_2020_learned_TESTpredict_dataset.pkl'))

# -----------------prediction where sr355 < 1.0-----------------
# predict_dataset = model.predict(test_dataset.iloc[np.where((test_dataset['sr355']<1.0) & (test_dataset['sr355']>0))[0]])
# pd.DataFrame(predict_dataset).to_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/','ipral_2020_learned_TESTpredict_dataset_SR355_1.0_left.pkl'))
# testtarget_dataset.iloc[np.where((test_dataset['sr355']<1.0) & (test_dataset['sr355']>0))[0]].to_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/','ipral_2020_learned_TESTtarget_dataset_SR355_1.0_left.pkl'))

# -----------------prediction where sr355 > 1.0-----------------
# predict_dataset = model.predict(test_dataset.iloc[np.where((test_dataset['sr355']>1.0))[0]])
# pd.DataFrame(predict_dataset).to_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/','ipral_2020_learned_TESTpredict_dataset_SR355_1.0_right.pkl'))
# testtarget_dataset.iloc[np.where((test_dataset['sr355']>1.0))[0]].to_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/','ipral_2020_learned_TESTtarget_dataset_SR355_1.0_right.pkl'))


# In[87]:


# params_plot = plots('LEARNING', 'IPRAL2020', np.arange(0.01,1,0.05), 0, 50, test_dataset.values[:,0], testtarget_dataset.values.ravel(), 
#                     f'{(~np.isnan(testtarget_dataset.values)).sum()} validated points total', ['355 mesures', '532 mesures'], ' ')
# params_plot.precent_hist2d()

predict_dataset = pd.read_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/', 'ipral_2018-2019-2020_learned_TESTpredict_dataset.pkl'))

plt.hist2d(testtarget_dataset.values.ravel(), predict_dataset.values.ravel(), norm=LogNorm(vmax=1e5), range=[[0,80], [0,80]], bins=100, density=False)
plt.colorbar()
plt.plot(np.arange(0,80,1)+2, np.arange(0,80,1), color='r')
plt.plot(np.arange(0,80,1)-2, np.arange(0,80,1), color='r')
plt.title('testtarget_dataset.values.ravel(), predict_dataset.values.ravel()')
plt.xlabel('532 mesures')
plt.ylabel('532 predict')
plt.title('LEARNING-IPRAL-2018-2019-2020')


# In[193]:



# plt.hist2d(testtarget_dataset.values.ravel(), predict_dataset, norm=LogNorm(), range=[[0,50], [0,50]], bins=100, density=True)
# plt.colorbar()
# plt.title('testtarget_dataset.values.ravel(), predict_dataset.values.ravel()')

params_plot = plots('LEARNING', 'IPRAL2020', np.arange(0.01,1,0.05), 0, 50, testtarget_dataset.values.ravel(), test_dataset.values[:,0]*5.3,
                    f'{(~np.isnan(testtarget_dataset.values)).sum()} validated points total', ['532 mesures', '532 predict'], ' ')
params_plot.precent_hist2d()


# ### QUANTIFY

# In[6]:


method = 'LEARNING'

listfiles = sorted(Path('/home/nmpnguyen/conversion_model/comparaison/').glob(f'ipral_2020-{method}-Stats-between-0.0-40.0*'))
listfiles 
labels = ['where 0 < SR355 < 1', 'where SR355 > 1', 'all SR355'] #
COLORS = ['g', 'r', 'b']


# In[58]:


method = 'THEORITICAL'
listfiles = sorted(Path('/homedata/nmpnguyen/comparaison/').glob('IPRAL2018_IPRAL2019_IPRAL2020-THEORITICAL*'))
print(listfiles[0].stem.split('-')[2], len(listfiles))


# In[60]:


# quantify


fig, ax = plt.subplots(figsize = (8,8))
for filepkl, l in zip(listfiles, range(len(labels))):
    data_pkl = pd.read_pickle(filepkl)
    ax.plot(data_pkl.index, data_pkl.values, label = f'{filepkl.stem.split("-")[2]}') #label = labels[l], color = COLORS[l]

colormap = plt.cm.turbo #nipy_spectral, Set1,Paired   
colors = [colormap(i) for i in np.linspace(0, 1,len(ax.lines))]
for i,j in enumerate(ax.lines):
    j.set_color(colors[i])

ax.legend(loc=2, title = 'Coefs')
ax.set_ylabel('Proportion of captured points \naround the diagonal (%)')
ax.set_xlabel('Unit around the diagonal')
ax.set_title(f'IPRAL_2020-{method} METHOD')
plt.show()


# In[91]:


data_pkl = pd.read_pickle(Path('/home/nmpnguyen/conversion_model/comparaison/2018IPRAL2019IPRAL2020-LEARNING-Stats-between-0.0-80.0-None-None.pkl'))

plt.plot(data_pkl.index, data_pkl.values)
plt.ylabel('Proportion of captured points \naround the diagonal (%)')
plt.xlabel('Unit around the diagonal')
plt.title(f'IPRAL_2018-2019-2020 LEARNING METHOD')


# In[6]:


class check:
    def __init__(self, min_value: float, max_value: float, closed_unit: float,
                 x_value: float, y_value: float):
        self.x1, self.y1 = min_value-closed_unit, min_value
        self.x2, self.y2 = min_value+closed_unit, min_value
        self.x3, self.y3 = max_value+closed_unit, max_value
        self.x4, self.y4 = max_value-closed_unit, max_value
        self.x, self.y = x_value, y_value
 
    # A function to check whether point P(x, y) lies inside the rectangle
    # formed by A(x1, y1), B(x2, y2), C(x3, y3) and D(x4, y4)
    def check_point(self):
        area = lambda X1, Y1, X2, Y2, X3, Y3 : abs((X1 * (Y2 - Y3) + X2 * (Y3 - Y1) + X3 * (Y1 - Y2)) / 2.0)
        # Calculate area of rectangle ABCD
        A = (area(self.x1, self.y1, self.x2, self.y2, self.x3, self.y3) + 
             area(self.x1, self.y1, self.x4, self.y4, self.x3, self.y3))

        # Calculate area of triangle PAB
        A1 = area(self.x, self.y, self.x1, self.y1, self.x2, self.y2)
    #     print('Aire PAB', A1)

        # Calculate area of triangle PBC
        A2 = area(self.x, self.y, self.x2, self.y2, self.x3, self.y3)
    #     print('Aire PBC', A2)

        # Calculate area of triangle PCD
        A3 = area(self.x, self.y, self.x3, self.y3, self.x4, self.y4)
    #     print('Aire PCD', A3)

        # Calculate area of triangle PAD
        A4 = area(self.x, self.y, self.x1, self.y1, self.x4, self.y4)
    #     print('Aire PAD', A4)

        # Check if sum of A1, A2, A3
        # and A4 is same as A
        # print('Aire PAB + PBC + PCD + PAD', np.round(A1 + A2 + A3 + A4, decimals=2))
        # print('Aire A', np.round(A, decimals=2))
        return (np.round(A, decimals=2) == np.round(A1 + A2 + A3 + A4, decimals=2))

    def quantify(self):
        points_checked = self.check_point()
        proportion_coef = 100*np.where(points_checked == True)[0].shape[0]/points_checked.shape[0]
        return proportion_coef


# In[223]:


p = check(0.0, 80.0, 1.0, sr532mes.ravel(),  sr532pred.ravel())
p.quantify()


# In[191]:


min_value = 0.0
max_value = 80
F = lambda x,y : check_point(min_value-unit,min_value,
                             min_value+unit,min_value,
                             max_value+unit,max_value,
                             max_value-unit,max_value,x,y)

# check_arr = F(sr532mes.ravel(), sr532pred.ravel())

print(sr532mes.ravel()[4318150], sr532pred.ravel()[4318150], F(sr532mes.ravel()[4318150], sr532pred.ravel()[4318150]))


# In[181]:


np.where(check_arr==True)[0].shape#, np.where((sr532mes.ravel()<1.5)&(sr532mes.ravel()>0.0)& (sr532pred.ravel()<1.5)& (sr532pred.ravel()>0.0))[0]
# x_edges[0:2], x_edges[-3:-1], y_edges[0], y_edges[-1]


# In[165]:


get_ipython().magic('matplotlib inline')
plt.plot(np.arange(0,80), np.arange(0,80), '-k')
plt.hist2d(sr532mes.ravel()[np.where(check_arr==True)[0]], sr532pred.ravel()[np.where(check_arr==True)[0]], 
           bins=200, range=[[0,80], [0,80]], norm=LogNorm(vmax=1e2))
plt.colorbar()
plt.plot(np.arange(0,80)+unit, np.arange(0,80), '-y')
plt.plot(np.arange(0,80)-unit, np.arange(0,80), '-y')
plt.grid()
plt.xlabel('sr532 mesures')
plt.ylabel('sr532 prediction')
plt.ylim(0,10)
plt.xlim(0,10)


# In[7]:


def proportion_validated(matrix_measured, matrix_predited, closed_unit, min_value, max_value):
    xedges = yedges = np.arange(min_value, max_value, closed_unit)
    F1 = lambda x,y: check_point(min_value-closed_unit, min_value,
                                 min_value+closed_unit, min_value,
                                 max_value+closed_unit, max_value,
                                 max_value-closed_unit, max_value,
                                 x,y)

    check_points = F1(matrix_measured.ravel(), matrix_predited.ravel())
    propt_valid = 100*np.where(points_valid==True)[0].shape[0]/check_points.shape[0]
    return propt_valid


# In[195]:


proportion_validated(sr532mes, sr532pred, unit, 0.0, 80.0)

# np.isin(points_valid, np.where(check_arr==True)[0])#/sr532mes.ravel().shape[0]


# In[172]:


maindir = '/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/2020/'
pattern = 'ipral_1a_Lz1R15mF30sPbck_v01_*_000000_1440.nc'

dataset355 = get_folder_dataset(maindir, pattern, ipral_characters, 355, True)

dataset355 = set_dataset_with_condition(dataset355, None)


# In[224]:


# dataset532 = get_folder_dataset(maindir, pattern, ipral_characters, 532, True)
dataset532mes = pd.read_pickle('/home/nmpnguyen/conversion_model/comparaison/ipral_dataset532mes.pkl')


# In[226]:


dataset532mes.shape


# ## Aerosols extinction coef & backscatter coef

# In[8]:


ns = 1.00028571
rho_n = 0.03
#-------------
Na = 6.02e23 #Avodrago
R = 8.31451
P = 101325 #Pa
T = 273.15 + 15 #K
numb_density = (P*Na)/(T*R)*1e-6
Nso = 2.54743e19
#-------------
W = 355e-7
import math
math.pi


# In[9]:


# Rayleigh backscattering cross section 
print(numb_density, Nso)
cross_section_355 = (24*(math.pi**3))/((W**4)*np.square(Nso)) * np.square((np.square(ns)-1)/(np.square(ns)+2)) * ((6+3*rho_n)/(6-7*rho_n))
print(cross_section_355)


# In[10]:


def molecular_numb_density_profil(P, T):
    # P in Pa, T in K
    Na = 6.02e23
    R = 8.31451
    numb_density = (P)/(T*R)
    return numb_density


# In[11]:


# pd_simul = pd.read_pickle('/homedata/nmpnguyen/IPRAL/RF/Simul/ipral_1a_Lz1R15mF30sPbck_v01_20201129_000000_1440_simul.pkl')
# pressure_file = pd_simul['pression']
# tempe_file = pd_simul['ta']

# pressure_file = pressure_file.unstack()
# tempe_file = tempe_file.unstack()

# time = pd_simul.reset_index(inplace=True)['time']
# pd_simul.shape


netcdf_simul = xr.open_dataset('/homedata/nmpnguyen/database_lidars/tmp_simul.nc')
pressure_file = netcdf_simul['pression']
tempe_file = netcdf_simul['ta']


# In[12]:


# print(pressure_file, tempe_file)
density_profil = molecular_numb_density_profil(pressure_file, tempe_file)
print(density_profil)


# In[13]:


ipraldt = xr.open_dataset(Path('/bdd/SIRTA/pub/basesirta/1a/ipral/2018/10/15/ipral_1a_Lz1R15mF30sPbck_v01_20181015_000000_1440.nc'))
ipraldt


# In[14]:


channel_Raman = 'rcs_14'
channel_355 = 'rcs_12'
rcs_Raman = (ipraldt[channel_Raman]/np.square(ipraldt['range']) - ipraldt['bckgrd_'+channel_Raman])#*np.square(ipraldt['range'])
rcs_355 = (ipraldt[channel_355]/np.square(ipraldt['range']) - ipraldt['bckgrd_'+channel_355])*np.square(ipraldt['range'])


# In[16]:


# (ipraldt[channel_355]/np.square(ipraldt['range'])).isel(time=0), ipraldt[channel_355].isel(time=0)


rcs_355.plot(x='time', y='range', norm=LogNorm(vmin=1e0), ylim=(0, 35000))


# In[32]:


rcs_355.sel(range = ipraldt.range[400:405]).mean(dim='range'), ipraldt.range[400:405]


# In[303]:


density_profil


# In[326]:


def extinction_aerosol(alt, n, rcs_raman, ext_mol_355, ext_mol_raman):
    wavelength = 355
    wavelength_raman = 387 
    derive = np.zeros_like(n)
    ext_aer_355 = np.zeros_like(n)
    print('Shape of derive and ext_aer', derive.shape, ext_aer_355.shape)
    for z in range(len(alt)-1):
        derive[:, z] = (-n[:,z+1]/rcs_raman[:, z+1] + n[:,z]/rcs_raman[:, z])/(alt[z+1] - alt[z])
        ext_aer_355[:, z] = (derive[:, z] - ext_mol_355[:, z] - ext_mol_raman[:, z])/(1 + 355/387)
    print(derive)
    return ext_aer_355

def backscatter_aerosol(SR, n, cross_section_Ray):
    R = SR - 1
    bcs_aer = R*n*cross_section_Ray
    return bcs_aer


# In[305]:


print(ipraldt['range'][1], density_profil.shape)
alpha_aer_355 = extinction_aerosol(ipraldt['range'].values, density_profil, (rcs_Raman*np.square(ipraldt['range'])).values, 
                                      netcdf_simul['alpha355'], netcdf_simul['alpha387'])


# In[310]:


A = density_profil/rcs_Raman
a = np.zeros_like(A)
for t in range(A.shape[1]-1):
    a[:, t] = (A[:, t+1] - A[:, t])/(ipraldt['range'][t+1] - ipraldt['range'][t])*1/A[:,t]


# In[313]:


alpha_aer_355 = (a - netcdf_simul['alpha355']- netcdf_simul['alpha387'])/(1 + 355/387)


# In[324]:


# plt.plot(alpha_aer_355[100,:], ipraldt['range'].values)
# plt.ylim(0,8000)

plt.pcolormesh(ipraldt['time'].values, ipraldt['range'].values, alpha_aer_355.T, norm=LogNorm(), shading='flat')
plt.colorbar(extend = 'both')
plt.ylim(0, 20000)


# In[341]:


maindir = '/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/2020/'
pattern = 'ipral_1a_Lz1R15mF30sPbck_v01_20200805_000000_1440.nc'


dt = xr.open_dataset(Path(maindir, pattern))
print(dt['time'])
# (dt['calibrated']/dt['simulated'] - 1)*density_profil*6.02e23*cross_section_355
bcs_aer_355 = backscatter_aerosol((dt['calibrated']/dt['simulated']).sel(wavelength=355), density_profil*6.02e23, cross_section_355)


# In[349]:


bcs_aer_355[:,:1333]#.plot(x='time', y='range', ylim=(0, 20000),cmap='turbo', vmin=-10, vmax=0)

