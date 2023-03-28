import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

'''
Ce script est pour créer le modèle Decision Tree et tester 
Ce script va appeler les fonctions de création des Data train et les Data test.
'''
from datetime import datetime
released_time = datetime.now().strftime("%d%m%Y_%H:%M:%S")

import sys
sys.path.append('/home/nmpnguyen/')
import train_data_DCT as train
# import test_data_DCT as test

#---------------------
#---------------------

# generate Decision Tree model and Apply test data
def Model_and_Test(Xinput, Yinput, Xtest, Ytest):
    '''
    Créer un model de Decision Tree avec 2 features, dont Xinput doit conserver 2 paramètres
    '''
    from sklearn.tree import DecisionTreeRegressor as DTR
    tree = DTR(min_samples_leaf=5)
    tree.fit(Xinput,Yinput)
    Y_pred = tree.predict(Xtest)
    Y_residus = Ytest - Y_pred
    return Y_pred, Y_residus

#---------------------
#---------------------

Listfiles = sorted(Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/zone-3000-4000').glob('ipral_1a_Lz1R15mF30sPbck_v01_2018*.nc'))

# Préparer les data pour train model 
print('Préparer les data pour train model')
train355_Z, train532= train.get_all_data_selected_2features(Listfiles[:50], 100, 14000)

fig, ax = plt.subplots()
h = ax.hist2d(train355_Z[:,0], train532, bins=100, range=[[-10, 40], [-10, 80]], norm=LogNorm())
plt.colorbar(h[3], ax=ax)
title = f'TRAIN DATA, 15m x 15mins \n{Listfiles[0].parents[0]}'
ax.set(xlabel='SR355_mesured', ylabel='SR532_mesured', title=title)
plt.savefig(f'/home/nmpnguyen/scatterplot_train_data_DCT_{released_time}.png')
plt.close()
plt.clf()

# Préparer les données Test 
# Multi profiles 
print('Préparer les données Test - Multi profiles ')
test355_Z, test532= train.get_all_data_selected_2features(Listfiles[50:], 100, 14000)

fig, ax = plt.subplots()
h = ax.hist2d(test355_Z[:,0], test532, bins=100, range=[[-10, 40], [-10, 80]], norm=LogNorm())
plt.colorbar(h[3], ax=ax)
title = f'TEST DATA, 15m x 15mins \n{Listfiles[0].parents[0]}'
ax.set(xlabel='SR355_mesured', ylabel='SR532_mesured', title=title)
plt.savefig(f'/home/nmpnguyen/scatterplot_test_data_DCT_{released_time}.png')
plt.close()
plt.clf()

    # Appliquer le model
print('Appliquer le model')
SR532_pred, SR532_residus = Model_and_Test(Xinput=train355_Z, Yinput=train532, Xtest=test355_Z, Ytest=test532)
DCT_describe = pd.DataFrame(SR532_residus).describe()

from pandas.plotting import table
fig, ax = plt.subplots()
h = ax.hist2d(test355_Z[:,0], SR532_pred, bins=100, range=[[-10, 40], [-10, 80]], norm=LogNorm())
plt.colorbar(h[3], ax=ax)
table(ax, np.round(DCT_describe, 3), loc="upper right", colWidths=[0.2, 0.2, 0.2]) # add table of describe into plot
title = f'PREDICTION DATA, 15m x 15mins \n{Listfiles[0].parents[0]}'
ax.set(xlabel='SR355_mesured', ylabel='SR532_mesured', title=title)
plt.savefig(f'/home/nmpnguyen/scatterplot_predicted_data_DCT_{released_time}.png')
plt.close()
plt.clf()

# Préparer les données Test 
# 1 profile
print('Préparer les données Test - 1 profile ')
n = random.sample(range(len(Listfiles[50:])), 1)
test355_Z, test532= train.get_all_data_selected_2features(Listfiles[n], 1, 14000)
    
    # Appliquer le model
print('Appliquer le model')
SR532_pred, SR532_residus = Model_and_Test(Xinput=train355_Z, Yinput=train532, Xtest=test355_Z, Ytest=test532)
DCT_describe = pd.DataFrame(SR532_residus).describe()

fig, ax = plt.subplots()
ax.plot(test355_Z[:,0], test355_Z[:,1], color='b', label='355')
ax.plot(test532[:,0], test355_Z[:,1], color='g', label='532')
ax.plot(SR532_pred[:,0], test355_Z[:,1], color='r', label='532-predict')
title = f'PREDICTION DATA, 15m x 15mins \n{Listfiles[n]}'
ax.set(xlabel='SR', ylabel='Height', title=title)
plt.savefig(f'/home/nmpnguyen/profile_predicted_data_DCT_{released_time}.png')
plt.close()
plt.clf()
