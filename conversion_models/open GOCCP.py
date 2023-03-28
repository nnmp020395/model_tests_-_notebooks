
# coding: utf-8

# In[3]:


import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


import math
def computeMolATB_Calipso(P, T):
    Ra = 8.314472
    Na = 6.02e23
    Qs = 5.167e-27
    k = 1.0401
    #P in hPa, T in K
    scat_vol = Na/Ra * Qs * P/T
    beta_vol = scat_vol/(8*math.pi*k/3)
    return scat_vol, beta_vol


def computeAttnMolATB_Calipso(alt, scat_vol, beta_vol):
    import math
    opt_depth = np.zeros_like(scat_vol)
    for i in range(1,(len(alt))):
        opt_depth[i] = scat_vol[i-1]+scat_vol[i]*(alt[i]-alt[i-1])
        
    AttnMolATB = beta_vol * np.exp(-2*opt_depth)
    return AttnMolATB


def computeRayleighScat(w):
    n_s2 = 5.5647e-4 + 1
    Nso = 2.547e19 
    DepFact = 0.0284
    w = w*1e-3
    RayScat = (24*(math.pi**3)*(n_s2-1)**2)/(w**4*Ns**2*(n_s2+2)**2) * (6+3*DepFact)/(6-7*DepFact)
    return RayScat


def computeTotalMolVolBscCoef(ND, RayScat):
    Sr = math.pi*8/3
    TotalMolVolBscCoef = ND*RayScat/Sr
    return TotalMolVolBscCoef


def ConstCalibration532(r_c, p_c, Eo, TotalMolVolBscCoef):
    ConstCal532 = (r_c**2)*p_c/(Eo*TotalMolVolBscCoef)
    return ConstCal532



# In[3]:


#_____Importer les fonctions______
import sys
sys.path.append('/home/nmpnguyen/')
from imp import reload as rl
import Attn_Bsc_theorie 
rl(Attn_Bsc_theorie)
from Attn_Bsc_theorie import Method1, Method3, Pression


# ### Description
# <ol>
# <li>E1: Ouvrir les fichiers GOCCP et CALIPSO </li>
# <li>E2: Lire les variables </li>
# GOCCP : mid_alt, time, ATB, ATB_mol, instant_SR, TEMP
# CALIPSO: Profile_UTC_Time, Total_Attenuated_Backscatter_532, Temperature, Pression / créer le dataset de l'Altitude pour les données de Calipso selon le "CALIPSO Data User’s Guide - Lidar Level 1 - Version 4.10"
# <li>E3: Import la méthode 1 et 3 pour calculer Attn. Molecular Backscatter </li>
# Transformer les unités 
# Normaliser Attn. Molecular Backscatter
# Plotter avec Attn Total Backscatter et trouver SR 
# </ol>

# ## GOCCP

# In[170]:


#_____E1________
goccpdir = Path('/bdd/CFMIP/CFMIP_OBS_LOCAL/GOCCP_v3/instant_SR_CR_DR/v3.1.2/2019/day/')
#Path('/bdd/CFMIP/CFMIP_OBS_LOCAL/GOCCP_v3/SR_histo/grid_1x1xL40/2019/day/daily/')


# In[171]:


goccppath = sorted(goccpdir.glob("instant_SR_CR_DR_2019-12-11*.nc"))
print(goccppath[0])


# In[185]:


go = xr.open_dataset(goccppath[0])
print(go.variables)
print('-------------')


# In[173]:


def fractionalTime2UTC(t):
    import math
    hour, date = math.modf(ca['Profile_UTC_Time'][-1])
    hour = hour*24
    mins = math.modf(hour)[0]*60
    
    


# In[210]:


#_______Calculer la pression pour GOCCP_______
go_PRES = Pression.eq_hydrostat(go['alt_mid']*1e3)
print(go_PRES[0:10])
#______Average + plot ATB, ATBmol__________ 
go_ATB_mean = go['ATB'][900:1100,:].mean(('it'))
go_ATBmol_mean = go['ATB_mol'][900:1100,:].mean(('it'))
go_TEMP_mean = go['TEMP'][900:1100,:].mean(('it'))+273.15
go_SR_mean = go['instant_SR'][900:1100,:].mean(('it'))

#______Méthode 3______
go_molext, go_molbsc = Method3.computeMolBackscatter(P = go_PRES, T = go_TEMP_mean, w=532)
go_ATBmol_syn = Method3.computeAttnMolBackscatter(extinction = go_molext, beta_vol = go_molbsc,
                                                  alt = go['alt_mid']*1e3, datafrom='top')
print(go_molext)

A = np.zeros_like(go_molext)
for i in range(len(go['alt_mid'])-2,-1,-1):
    A[i] = A[i+1] - go_molext[i]*(go['alt_mid'][i+1]-go['alt_mid'][-1])*1e3

B = go_molbsc*np.exp(-2*A)
print(B)
#normaliser 
zbottom = 15
ztop = 17.5
idx = (go['alt_mid']>=zbottom) & (go['alt_mid']<=ztop)
print(np.mean((go_ATB_mean[idx]*1e-3)/go_ATBmol_syn[idx]))
c = np.mean((go_ATB_mean[idx]*1e-3)/go_ATBmol_syn[idx])
go_ATBmol_syn_norm = go_ATBmol_syn*c
print(c)
c = np.mean((go_ATB_mean[idx]*1e-3)/go_molbsc[idx])
go_molbsc_norm = go_molbsc*c
print(c)


# In[211]:


fig, ax = plt.subplots(figsize=[10,6])
#ax.semilogx(go_ATB_mean*1e-3, go['alt_mid'], label='ATB, 1H averahe profile', color='b')
#ax.semilogx(go_ATBmol_mean*1e-3, go['alt_mid'], label='ATBmol, 1H averahe profile', color='k', linestyle='--')
#ax.semilogx(go_ATBmol_syn, go['alt_mid'], label='Attn.ATBmol.norm by Method 1', color='g')
ax.semilogx(go_molbsc, go['alt_mid'], label='ATBmol.norm by Method 1', color='r')
ax.semilogx(B, go['alt_mid'], label='Attn.ATBmol by Method 1', color='g')
#ax.fill_betweenx(go['alt_mid'], go_ATBmol_syn_norm-rmse, go_ATBmol_syn_norm+rmse, 
#                 alpha=0.25, color='g', label='rmse of ATBmol by Method 1')
ax.axhspan(zbottom, ztop, color='y', alpha=0.5, lw=0, label='normalization height')
leg = ax.legend()
leg.set_title('GOCCP')
ax.set(ylabel='Altitude, km', xlabel='Attn. Backscatter, 1/m/sr ')
#ax.set_xlim(0, 0.5e-5)



#ax2.plot(go_SR_mean, go['alt_mid'], label='GOCCP SR', color='b')
#ax2.plot((go_ATBmol_mean*1e-3)/go_ATBmol_syn_norm, go['alt_mid'], label='SR from ATBmol by Method 1', color='g')


# In[53]:


fig2, ax2 = plt.subplots(figsize=[10,6])
ax2.plot(go['instant_SR'].isel(it=1000), go['alt_mid'], label='GOCCP SR', color='b')
ax2.plot(go['ATB'].isel(it=1000)/go['ATB_mol'].isel(it=1000), go['alt_mid'], color='g')
ax2.vlines(1, ymin=go['alt_mid'].values.min(), ymax=go['alt_mid'].values.max(), color='k')
ax2.set_xlim(-0.5, 5)
ax2.legend()
ax2.set(xlabel='Scattering Ratio')


# In[177]:


ca_MND = compute_numb_density(P=ca['Pressure'][100,:].values*100, T=ca['Temperature'][100,:].values+273.15)
print(ca_MND.shape)
fig, ax = plt.subplots()
ax.plot(ca_MND, met_alt, label='theorical')
ax.plot(ca['Molecular_Number_Density'][100,:].values, met_alt, label='calipso')
ax.legend()
ax.set(xlabel='Molecular_Number_Density \nmolecules par m3')


# In[90]:


#RayScat532 = computeRayleighScat(532)
#bcsmol532 = computeTotalMolVolBscCoef(ND=ca['Molecular_Number_Density'][100,:], RayScat=RayScat532)

# non calculer Constance de calibration, mais calculer R = ATB/ARB mol 

mol532_Calipso = computeMolATB_Calipso(P=ca['Pressure'][100,:].values, T=ca['Temperature'][100,:].values+273.15)
print(mol532_Calipso[1])
print(ca['Total_Attenuated_Backscatter_532'][100,:])
Attnmol532_Calipso = computeAttnMolATB_Calipso(alt=met_alt, scat_vol, beta_vol)

#attnmol532_Calipso = 
fig, ax = plt.subplots()
ax.plot(mol532_Calipso[1], met_alt)
ax2 = ax.twiny()
ax2.plot(ca['Total_Attenuated_Backscatter_532'][100,1:], alt_range[1:], color='r')
ax.set_ylim(20,40)


# In[180]:


mol532_C = mol532_Calipso[1][((met_alt>=35)&(met_alt<=37.5))].mean()
print(mol532_C)
attn_bsc532_C = ca['Total_Attenuated_Backscatter_532'][100,((alt_range>=35)&(alt_range<=37.5))].mean()
print(attn_bsc532_C)
ConstCal = attn_bsc532_C/mol532_C
print(ConstCal)

fig, ax = plt.subplots()
ax.hlines(35, xmin=-0.0001, xmax=0.001, color='k')
ax.hlines(37.5, xmin=-0.0001, xmax=0.001, color='k')
ax.plot(mol532_Calipso[1], met_alt, zorder=10, label='theoritical')
ax.plot(ca['Total_Attenuated_Backscatter_532'][100,1:]/ConstCal, alt_range[1:], color='r', label='calipso')
ax.set_xlim(-0.0001,0.001)
ax.legend()


# In[338]:


#_____test 2nd theoritical method___: alpha(N, P, T, wave)
import math
def compute_numb_density(P,T):
    # P in Pa, T in K
    Na = 6.02e23
    R = 8.31451
    numb_density = (P*Na)/(T*R)
    return numb_density


def computeMolExtinction(P, T, ND):
    DepFactor = 0.0284
    n_s2 = 5.5647e-4+1
    Nso = 2.547e19 #cm-3
    wave = 532e-3
    T0 = 288.15 #K
    P0 = 1013.25 #hPa 
    MolExtinctionConst = (8*(math.pi)**3/(3*wave**4))*((n_s2-1)/Nso)**2*((6+3*DepFactor)/(6-7*DepFactor))*T0/P0
    MolExtinction = MolExtinctionConst*ND*P/T
    return MolExtinction

def computeMolBackscatter(MolExtinction):
    MolBackscatter = MolExtinction/(8*math.pi/3)    
    return MolBackscatter

def computeAttnMolBsc(beta_vol, scat_vol, alt):
    opt_depth = np.zeros_like(scat_vol)
    for i in range(1,(len(alt))):
        opt_depth[i] = opt_depth[i-1]+scat_vol[i]*(alt[i]-alt[i-1])
        
    AttnMolATB = beta_vol * np.exp(-2*opt_depth)
    return AttnMolATB


# In[341]:


#_____CALIPSO_____
calipsopath = list(calipsodir.glob('CAL_LID_L1-Standard-V4-10.*ZN.hdf'))
ca = xr.open_dataset(calipsopath[0])

ca_MND = compute_numb_density(P = ca['Pressure'][100,:].values*100, T = ca['Temperature'][100,:].values+273.15)
ca_Ext = computeMolExtinction(P=ca['Pressure'][100,:].values, T=ca['Temperature'][100,:].values+273.15, ND=ca_MND)
ca_BscMol = computeMolBackscatter(ca_Ext)
ca_AttnBscMol = computeAttnMolBsc(ca_BscMol, ca_Ext, met_alt)
print(ca_AttnBscMol)


# In[346]:


fig, ax = plt.subplots()
#ax.plot(mol532_Calipso[1], met_alt)
ax.plot(ca_AttnBscMol, met_alt)
ax2 = ax.twiny()
ax2.plot(ca['Total_Attenuated_Backscatter_532'][100,1:], alt_range[1:], color='r')
ax.set_ylim(20,40)


# In[347]:


mol532_C = ca_AttnBscMol[((met_alt>=35)&(met_alt<=37.5))].mean()
print(mol532_C)
attn_bsc532_C = ca['Total_Attenuated_Backscatter_532'][100,((alt_range>=35)&(alt_range<=37.5))].mean()
print(attn_bsc532_C)
ConstCal = attn_bsc532_C/mol532_C
print(ConstCal)

fig, ax = plt.subplots()
#ax.hlines(35, xmin=-0.0001, xmax=0.001, color='k')
#ax.hlines(37.5, xmin=-0.0001, xmax=0.001, color='k')
ax.plot(ca_AttnBscMol, met_alt, zorder=10, label='theoritical')
ax.plot(ca['Total_Attenuated_Backscatter_532'][100,1:]/ConstCal, alt_range[1:], color='r', label='calipso')
#ax.set_xlim(-0.0001,0.001)
ax.legend()


# ## CALIPSO

# In[4]:


calipsodir = Path('/bdd/CALIPSO/Lidar_L1/CAL_LID_L1.v4.10/2019/2019_12_11/')#CAL_LID_L1-Standard-V4-10.2016-07-14T01-13-03ZN.hdf')
calipsopath = sorted(calipsodir.glob('CAL_LID_L1-Standard-V4-10.*ZN.hdf'))
print(calipsopath[0])


# In[7]:


ca = xr.open_dataset(calipsopath[0])
print(ca.variables)


# In[9]:


#_______Calculer l'altitude de Calipso selon le guide de l'index______

met_alt = np.linspace(40.0, -2.0, 33)
r_c = (met_alt>=30)&(met_alt>=34)
# Lidar_Data_Altitude
# This field defines the lidar data altitudes (583 range bins) to which lidar Level 1 profile products are registered.
alt_range = np.zeros([583])
alt_range[:33] = np.linspace(40.0, 30.1, len(alt_range[:33]))
alt_range[33:88] = np.linspace(30.1, 20.2, len(alt_range[33:88]))
alt_range[88:288] = np.linspace(20.2, 8.3, len(alt_range[88:288]))
alt_range[288:578] = np.linspace(8.3, -0.5, len(alt_range[288:578]))
alt_range[578:] = np.linspace(-0.5, -2.0, len(alt_range[578:]))

print(met_alt)


# In[43]:


#______Plot QL______
fig, ax = plt.subplots()
ca['Total_Attenuated_Backscatter_532'][1000:3000,:].plot(x='fakeDim60', y='fakeDim61', cmap='viridis', robust=True)


#______Average + plot ATB, ATBmol 
ca_ATB_mean = ca['Total_Attenuated_Backscatter_532'][1000:1800,:].mean(('fakeDim60'))
ca_TEMP_mean = ca['Temperature'][1000:1800,:].mean(('fakeDim84'))
ca_PRES_mean = ca['Pressure'][1000:1800,:].mean(('fakeDim86'))#milibars
ca_MND_mean = ca['Molecular_Number_Density'][1000:1800,:].mean(('fakeDim80'))

print(ca['Pressure'][1000:1800,:])


# In[64]:


#_______METHODE 3_________
# ca_extmol, ca_bcsmol  = Method3.computeMolBackscatter(P = ca_PRES_mean.values.astype('float'), 
#                                                 T = ca_TEMP_mean.values.astype('float')+273.15, 
#                                                 w=532)

# ca_ATBmol_syn3 = Method3.computeAttnMolBackscatter(alt = met_alt*1e3, extinction = ca_extmol, beta_vol = ca_bcsmol, datafrom = 'top')

import math
#ca_bcsmol = 5.167e-27*ca_MND_mean*(3/(8*math.pi))
#ca_extmol = ca_bcsmolsmol/0.119
#ca_ATBmol_syn1 = Method3.computeAttnMolBackscatter(alt = met_alt*1e3, extinction = ca_extmol, beta_vol = ca_bcsmol, datafrom = 'top')

#__________
Na = 6.02214e23
Ra = 8.314472
ca_scat = (Na*ca_PRES_mean.values*5.167e-27)/(ca_TEMP_mean.values*Ra)
print(ca_scat.shape)
ca_beta = ca_scat/(8*math.pi*1.0401/3)
tau = np.zeros_like(ca_scat)
for i in range(1,len(met_alt)):
    print((met_alt[i]-met_alt[0])*1e3)
    tau[i] = ca_scat[i]*(met_alt[i]-met_alt[0])*1e3 + ca_scat[0]
    
tau[0] = ca_scat[0]    
print((-2*tau))
ca_attnbeta = ca_beta*np.exp(-2*tau)
ca_Cpar = ca_attnbeta/(1-1/ca_beta)

# print(ca_Cpar)
# print(ca_attnbeta)
# print(ca_scat)
# print(met_alt)
# fig, ax = plt.subplots()
# ax.plot(ca_attnbeta[:30], met_alt[:30], label='attn', color='r')
# ax2 = ax.twiny()
# ax2.plot(ca_beta[:30], met_alt[:30], label='beta')


# In[40]:


#______Normalisation_______
zbottom = 30
ztop = 34
idx_range = (alt_range>=zbottom) & (alt_range<=ztop)
idx_met = (met_alt>=zbottom) & (met_alt<=ztop)

c1 = np.mean(ca_ATB_mean[idx_range]*1e-3)/np.mean(ca_attnbeta[idx_met])
#print(ca_ATBmol_syn)
print(c1)
ca_attnbeta_norm = ca_attnbeta*c1.values
print(ca_attnbeta_norm)

# c3 = np.mean(ca_ATB_mean[idx_range]*1e-3)/np.mean(ca_ATBmol_syn3[idx_met])
# #print(ca_ATBmol_syn)
# print(c3)
# ca_ATBmol_syn3_norm = ca_ATBmol_syn3*c1.values
# print(ca_ATBmol_syn3_norm)


# In[41]:


fig, ax = plt.subplots()
ax.semilogx(ca_attnbeta_norm, met_alt, label='Attn.Mol ATB', color='r')
ax.semilogx(ca_ATB_mean*1e-3, alt_range, label='Calipso Attn.ATB', color='b')
ax.axhspan(zbottom, ztop, color='y', alpha=0.5, lw=0, label='calibration height')
ax.set_xlim(0,2e-6)
ax.legend()
#ax2 = ax.twiny()
#ax2.plot(ca_ATB_mean*1e-3, alt_range, label='Attn.ATB')


# In[59]:


ca.variables


# In[65]:


ca['Total_Attenuated_Backscatter_532'][1000,:]
ca['Perpendicular_Attenuated_Backscatter_532'][1000,:] + ca['Parallel_Attenuated_Backscatter_532'][1000,:]

