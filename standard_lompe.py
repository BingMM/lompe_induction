#%% Import

import numpy as np
import pandas as pd
from datetime import timedelta
from lompe.utils.conductance import hardy_EUV
import apexpy
import matplotlib.pyplot as plt
import lompe
import scipy
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices
from tqdm import tqdm
import pickle

#%% Steps 

'''
Step 1: Run standard Lompe
Step 2: Save models
Step 3: Make plots
'''

#%% Inline fun

# Extracting a subsection of the data
def prepare_data(t0, t1):
    """ get data from correct time period """
    # prepare ampere
    amp = ampere[(ampere.time >= t0) & (ampere.time <= t1)]
    B = np.vstack((amp.B_e.values, amp.B_n.values, amp.B_r.values))
    coords = np.vstack((amp.lon.values, amp.lat.values, amp.r.values))
    f = grid.ingrid(coords[0, :], coords[1, :])
    B = B[:, f]
    coords = coords[:, f]
    amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', error = 30e-9, iweight=1.0)

    # prepare supermag
    sm = supermag[t0:t1]
    B = np.vstack((sm.Be.values, sm.Bn.values, sm.Bu.values))
    coords = np.vstack((sm.lon.values, sm.lat.values))
    f = grid.ingrid(coords[0, :], coords[1, :])
    B = B[:, f]
    coords = coords[:, f]
    sm_data = lompe.Data(B * 1e-9, coords, datatype = 'ground_mag', error = 10e-9, iweight=1.0)

    # prepare superdarn
    sd = superdarn.loc[(superdarn.index >= t0) & (superdarn.index <= t1)]
    vlos = sd['vlos'].values
    coords = np.vstack((sd['glon'].values, sd['glat'].values))
    los  = np.vstack((sd['le'].values, sd['ln'].values))
    f = grid.ingrid(coords[0, :], coords[1, :])
    vlos = vlos[f]
    los = los[:, f]
    coords = coords[:, f]
    sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', error = 50, iweight=1.0)
    
    return amp_data, sm_data, sd_data

#%% Paths

path_in  = '/home/bing/Dropbox/work/code/repos/lompe/examples/sample_dataset/'
path_out = '/home/bing/Dropbox/work/code/repos/lompe_induction/'

#%% Load data

# Data filename
supermagfn  = path_in + '20120405_supermag.h5'
superdarnfn = path_in + '20120405_superdarn_grdmap.h5'
iridiumfn   = path_in + '20120405_iridium.h5'

# load ampere, supermag, and superdarn data from 2012-04-05
ampere    = pd.read_hdf(iridiumfn)
supermag  = pd.read_hdf(supermagfn)
superdarn = pd.read_hdf(superdarnfn)

times = pd.date_range('2012-04-05 00:00', '2012-04-05 23:59', freq = '3Min')[:20] # Limiting testing to the first 100, instead of all 480 times
DT    = timedelta(seconds = 2 * 60) # will select data from +- DT

#%% Define grid

position = (-90, 65) # lon, lat
orientation = (-1, 2) # east, north
L, W, Lres, Wres = 4200e3, 7000e3, 100.e3, 100.e3 # dimensions and resolution of grid
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = 6481.2e3)

s_limit = np.min([grid.Wres, grid.Lres])/2

#%% Hyper parameters

Kp = 4 # for Hardy conductance model

#%% Step 1 + 3: Standard Lompe inversion and plot

ms = []

apex = apexpy.Apex(2012, refh = 110)
for i, t in tqdm(enumerate(times), total=times.size):
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')
    
    if i == 0:
        model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))
    else:
        model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    model.add_data(amp_data, sm_data, sd_data)
    gtg, ltl = model.run_inversion(l1 = 2, l2 = 0, save_matrices=True)

    plt.ioff()
    savepath = path_out + 'figures/standard_lompe/'
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')
    plt.ion()
    
    ms.append(model.m)

#%% Step 2: Save models

with open(path_out + '/data/standard_lompe_model_vectors.pkl', 'wb') as f:
    pickle.dump(ms, f)
