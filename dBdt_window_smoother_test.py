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

#%% Steps 

'''
Step 1: SECS of all time steps
Step 2: dBr/dt
Step 3: E induction
Step 4: Post-process
Step 5:
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

path_in  = '/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/standard_lompe/lompe/examples/sample_dataset/'
path_out = '/home/bing/BCSS-DAG Dropbox/Michael Madelaire/work/code/repos/lompe_induction/'

#%% Load data

# Data filename
supermagfn  = path_in + '20120405_supermag.h5'
superdarnfn = path_in + '20120405_superdarn_grdmap.h5'
iridiumfn   = path_in + '20120405_iridium.h5'

# load ampere, supermag, and superdarn data from 2012-04-05
ampere    = pd.read_hdf(iridiumfn)
supermag  = pd.read_hdf(supermagfn)
superdarn = pd.read_hdf(superdarnfn)

times = pd.date_range('2012-04-05 00:00', '2012-04-05 23:59', freq = '3Min') # Limiting testing to the first 100, instead of all 480 times
DT    = timedelta(seconds = 2 * 60) # will select data from +- DT

#%% Define grid

position = (-90, 65) # lon, lat
orientation = (-1, 2) # east, north
L, W, Lres, Wres = 4200e3, 7000e3, 100.e3, 100.e3 # dimensions and resolution of grid
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = 6481.2e3)

s_limit = np.min([grid.Wres, grid.Lres])/2

#%% Hyper parameters

Kp = 4 # for Hardy conductance model

#%% Step 1: SECS of all time steps

SECS_ms = [0]*times.size
Bu_pred = np.zeros((grid.shape[0], grid.shape[1], times.size))

_, _, Gu_pred = get_SECS_B_G_matrices(grid.lat.flatten(), grid.lon.flatten(), 
                                      6371.2*1e3, grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), 
                                      singularity_limit = s_limit)

# SECS fit
for i, t in tqdm(enumerate(times), total=len(times)):
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)

    f = grid.ingrid(sm_data.coords['lon'], sm_data.coords['lat'])

    Ge, Gn, Gu = get_SECS_B_G_matrices(sm_data.coords['lat'][f], sm_data.coords['lon'][f], 
                                       6371.2*1e3, grid.lat_mesh.flatten(), grid.lon_mesh.flatten(),
                                       singularity_limit = s_limit)
    G = np.vstack((Ge, Gn, Gu))
    
    d = sm_data.values[:, f].flatten()
    
    GTG = G.T.dot(G)
    GTd = G.T.dot(d)
    gtgmag = np.median(abs(np.diag(GTG)))
    
    m = np.linalg.solve(GTG + 5e0*gtgmag*np.eye(GTG.shape[0]), GTd)
    SECS_ms[i] = m
    
    Bu_pred[:, :, i] = Gu_pred.dot(m).reshape(grid.shape)

#%% Step 1: SECS of all time steps - Plot reconstruction
    
plt.ioff()
vmax = np.max(abs(Bu_pred))
clvls = np.linspace(-vmax, vmax, 20)
    
fig = plt.figure(figsize=(10,10))
    
amp_data, sm_data, sd_data = prepare_data(times[0] - DT, times[0] + DT)
f = grid.ingrid(sm_data.coords['lon'], sm_data.coords['lat'])
xi, eta = grid.projection.geo2cube(sm_data.coords['lon'][f], sm_data.coords['lat'][f])
plt.plot(xi, eta, '.', markersize=10, markerfacecolor='magenta', markeredgecolor='k', zorder=3)
    
for cl in grid.projection.get_projected_coastlines():
    xi, eta = cl
    plt.plot(xi, eta, linewidth=2, color='k', zorder=0)
    plt.plot(xi, eta, linewidth=1, color='cyan', zorder=1)

for i, t in tqdm(enumerate(times), total=len(times)):
    
    cc = plt.contourf(grid.xi, grid.eta, Bu_pred[:, :, i]*1e9, levels=clvls*1e9, cmap='bwr', zorder=-1)
    if i == 0:
        cax = plt.colorbar(cc)
        cax.set_label('Bu [nT]', fontsize=16)
    
        plt.xlim([grid.xi_min, grid.xi_max])
        plt.ylim([grid.eta_min, grid.eta_max])
        plt.xlabel('xi', fontsize=16)
        plt.ylabel('eta', fontsize=16)
        plt.gca().set_aspect('equal')
    
    plt.title('SECS reconstruction of Bu using Supermag: t={}'.format(i), fontsize=18)
    plt.savefig(path_out + 'figures/SECS_Bu/{}.png'.format(i), bbox_inches='tight')
    
    for c in cc.collections:
        c.remove()
    
plt.close('all')
plt.ion()

#%% Step 2: dBr/dt

dBudt_pred = np.diff(Bu_pred, axis=2, prepend=Bu_pred[:, :, -1:])[:, :, 1:]

#%% Step 3: E induction - Standard Lompe inversion for comparison and get relevant matrices

GsBu = []
Gs = []
ws = []
ds = []

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
    
    GsBu.append(model._B_df_matrix(grid.lat.flatten(), grid.lon.flatten(), 6371.2*1e3)[2*grid.size:, :])

    Gs.append(model._G)
    ws.append(model._w)
    ds.append(model._d)

#%% Step 3: E induction

w_size = 3
ms = np.zeros((grid.xi_mesh.size, times.size))

for i, t in tqdm(enumerate(times), total=times.size):
    
    j_mid = w_size + 0 # The window is 2*w_size+1 wide.
    
    i_start = i - w_size # The first timestep 
    i_stop = i + w_size # The last timestep
    
    # Ensure first timestep cannot be smaller than 0
    if i_start < 0:
        j_mid += i_start # If, e.g., -1 set to 0 and move j_mid down by 1.
        i_start = 0
    
    # Ensure last timestep does not exceed times.size-1.
    if i_stop > (times.size-1):
        j_mid -= i_stop - times.size-1 # If, e.g., 2 over max set to max and move j_mid down by 2.
        i_stop = times.size-1 # 1 is added later as last index is exclusive. Silly Python.
    
    nt = i_stop-i_start+1 # Number of timesteps included in this window
    
    # Generate matrices - Steady state
    G_ss = scipy.linalg.block_diag(*Gs[i_start:i_stop+1])
    d_ss = np.hstack(ds[i_start:i_stop+1])
    w_ss = np.hstack(ws[i_start:i_stop+1])

    # Generate matrices - Temporal
    G_t = np.zeros(((nt-1)*grid.size, nt*grid.xi_mesh.size)) # Allocate space
    G_t[:, 0:(nt-1)*grid.xi_mesh.size]             = - scipy.linalg.block_diag(*GsBu[i_start:i_stop])
    G_t[:, grid.xi_mesh.size:nt*grid.xi_mesh.size] = scipy.linalg.block_diag(*GsBu[i_start+1:i_stop+1])
    
    d_t = dBudt_pred[:, :, i_start:i_stop].flatten()
    
    w_t = np.ones(d_t.size) / (1*1e-9)**2    

    # Combine matrices
    G = np.vstack((G_ss, G_t))
    d = np.hstack((d_ss, d_t))
    w = np.hstack((w_ss, w_t))

    # Solve inverse problem
    GTG = G.T.dot(np.diag(w)).dot(G)
    GTd = G.T.dot(np.diag(w)).dot(d)

    gtgmag = np.median(abs(np.diag(GTG)))

    m = scipy.linalg.solve(GTG + 2e0*gtgmag*np.eye(GTG.shape[0]), GTd)
    ms[:, i] = m.reshape((nt, grid.xi_mesh.size)).T[:, j_mid]
    
#%% Step 4: Visualization

Bu_org = np.zeros((grid.xi_mesh.size, times.size))
Bu_new = np.zeros((grid.xi_mesh.size, times.size))

apex = apexpy.Apex(2012, refh = 110)
plt.ioff()
for i, t in tqdm(enumerate(times)):
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')
    
    if i == 0:
        model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))
    else:
        model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    model.add_data(amp_data, sm_data, sd_data)
    gtg, ltl = model.run_inversion(l1 = 2, l2 = 0)

    savepath = path_out + 'figures/lompe/'
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    
    _, _, Bu = model.B_ground()
    Bu_org[:, i] = Bu
    
    model.m = ms[:, i]
    savepath = path_out + 'figures/lompe_dbdt/'
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    
    _, _, Bu = model.B_ground()
    Bu_new[:, i] = Bu
    
    plt.close('all')
plt.ion()

#%%

dBu_org = np.zeros((Bu_org.shape[0], Bu_org.shape[1]-1))
dBu_new = np.zeros((Bu_org.shape[0], Bu_org.shape[1]-1))
for i in range(dBu_org.shape[1]):
    dBu_org[:, i] = Bu_org[:, i+1] - Bu_org[:, i]
    dBu_new[:, i] = Bu_new[:, i+1] - Bu_new[:, i]

vmax = np.max(abs(np.vstack((dBu_org, dBu_new))))
vmax = np.max([vmax, np.max(abs(dBudt_pred))])

clvls = np.linspace(-vmax, vmax, 40)

plt.ioff()
for i in range(times.size-1):
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].contourf(grid.xi_mesh, grid.eta_mesh, dBu_org[:, i].reshape(grid.xi_mesh.shape), levels=clvls, cmap='bwr')
    axs[1].contourf(grid.xi_mesh, grid.eta_mesh, dBu_new[:, i].reshape(grid.xi_mesh.shape), levels=clvls, cmap='bwr')
    axs[2].contourf(grid.xi, grid.eta, dBudt_pred[:, :, i], levels=clvls, cmap='bwr')
    plt.savefig(path_out + 'figures/dBu_comp/{}.png'.format(i), bbox_inches='tight')
    plt.close('all')
plt.ion()

#%%

plt.ioff()
fig, axs = plt.subplots(4, 2, figsize=(10,15))

for (ax, row, col) in zip(axs.flatten(),
                          [12, 12, 24, 24, 36, 36, 48, 48],
                          [12, 25, 12, 25, 12, 25, 12, 25]):
    ax.plot(times[:-1], dBu_org[row*37+col, :]*1e9)
    ax.plot(times[:-1], dBu_new[row*37+col, :]*1e9)
    ax.plot(times[:-1], dBudt_pred[row, col, :]*1e9)

plt.savefig(path_out + 'figures/time_series_comparison.png', bbox_inches='tight')
plt.close('all')
plt.ion()
