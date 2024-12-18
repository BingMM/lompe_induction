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
Step 1: Prepare some matrices that remain constant (I think)
Step 2: Create initial conditions
Step 3: Run Kalman filter
Step 4: Same filtered models
Step 5: Plot and save Lompe plots
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

#%% Step 1: Prepare constant matrices - General

# Time
nt = times.size
dt = 3*60

# Define conductance functions to be called in loop and set-up Lompe model
SH_fun = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall')
SP_fun = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

# Set-up lompe model for EZ access to various matrices
model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH_fun, SP_fun))

# Model dimensions
n_CF    = grid.xi_mesh.size # Number of model parameters describing Epot
n_DF    = grid.xi_mesh.size # Number of model parameters describing Eind
n       = n_CF + n_DF

#%% Step 1: Prepare Dynamic model function (DF)

A_DF = np.hstack((np.zeros((n_DF, n_CF)), np.eye(n_DF)))

#%% Step 1: Prepare Dynamic model function (CF)

# Br from CF E
## DF SECS design matrix for Br at ionosphere
_, _, Hu = get_SECS_B_G_matrices(grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), model.R, 
                                 grid.lat.flatten(), grid.lon.flatten(),
                                 current_type = 'divergence_free',
                                 RI = model.R,
                                 singularity_limit = model.secs_singularity_limit)

## QA 
QiA = model.QiA

## HQiA
HQiA_u = Hu @ QiA

## c_J necessary pieces
'''
c_J_CF =    - self.Dn.dot(SP) * Ee + self.De.dot(SP) * En \
            - self.Dn.dot(SH) * En * self.hemisphere \
            - self.De.dot(SH) * Ee * self.hemisphere \
            - SH * self.Ddiv.dot(E) * self.hemisphere
'''
hemisphere = model.hemisphere
Ee, En = model.Ee, model.En
E = np.vstack((Ee, En))
Dn = model.Dn
De = model.De
Ddiv = model.Ddiv
DdivdotE = Ddiv.dot(E)

def calc_G_r_CF(SH, SP):
    c_J_CF = - Dn.dot(SP) * Ee + De.dot(SP) * En \
             - Dn.dot(SH) * En * hemisphere - De.dot(SH) * Ee * hemisphere \
             - SH * DdivdotE * hemisphere
    G_r_CF = HQiA_u.dot(c_J_CF)
    return G_r_CF

# dB/dt from DF E
_, _, A_E = model.grid_E.projection.differentials(model.grid_E.xi, model.grid_E.eta,
                                                  model.grid_E.dxi, model.grid_E.deta, R=model.R)
A_E = np.diag(np.ravel(A_E))
Q_E = np.eye(model.grid_E.size) - A_E.dot(np.full((model.grid_E.size, model.grid_E.size), 1 / (4 * np.pi * model.R**2)))
c_E = -dt * np.linalg.inv(A_E) @ Q_E

'''
c_E = np.diag(-dt * np.linalg.inv(model.A) @ model.Q)
RBF = scipy.interpolate.Rbf(grid.xi.flatten(), grid.eta.flatten(), c_E)
c_E = RBF(grid.xi_mesh.flatten(), grid.eta_mesh.flatten())
c_E = np.diag(c_E)
'''

# T (model variance)    
def calc_T(G_r_CF, C_i):
    T =   G_r_CF @ C_i[:n_CF, :n_CF] @ G_r_CF.T \
        + c_E @ C_i[n_CF:, n_CF:] @ c_E.T
    return T

def calc_dmf_CF(SH, SP, C_i):
    G_r_CF  = calc_G_r_CF(SH, SP)
    T       = calc_T(G_r_CF, C_i)
    T       = np.eye(T.shape[0])
    T_inv   = np.linalg.pinv(T)
    del T
    
    GTT     = G_r_CF.T @ T_inv
    del T_inv

    GTTG    = GTT @ G_r_CF
    del G_r_CF
    
    GTTcE   = GTT @ c_E
    del GTT
    
    gtgmag = np.median(np.diag(GTTG))
    #A_CF = np.linalg.solve(GTTG + 2*gtgmag*np.eye(GTTG.shape[0]), np.hstack((GTTG, GTTcE)))
    A_CF = np.linalg.solve(GTTG, np.hstack((GTTG, GTTcE)))
    #A_CF = np.linalg.lstsq(GTTG + 1e-2*gtgmag*np.eye(GTTG.shape[0]), np.hstack((GTTG, GTTcE)), rcond=None)[0]
    #A_CF = np.linalg.lstsq(GTTG + 1e0*gtgmag*np.eye(GTTG.shape[0]), np.hstack((GTTG, GTTcE)), rcond=None)[0]
    #A_CF = np.linalg.lstsq(GTTG + 1e1*gtgmag*np.eye(GTTG.shape[0]), np.hstack((GTTG, GTTcE)), rcond=None)[0]
    del GTTG, GTTcE
    
    return A_CF

#%% Step 1: Prepare Dynamic model function

def calc_dmf(SH, SP, C_i):
    A_CF = calc_dmf_CF(SH, SP, C_i)
    A = np.vstack((A_CF, A_DF))
    return A

#%% Step 2: Initial conditions

i = 0
t = times[0]
model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
model.add_data(amp_data, sm_data, sd_data)
gtg, ltl = model.run_inversion(l1 = 2, l2 = 0, save_matrices=True)

#with open(path_out + '/data/standard_lompe_model_vectors.pkl', 'rb') as f:
#    ms_standard = pickle.load(f)

# Initial guess - Mean
#m0_CF   = np.zeros(n_CF) # Initial model of Epot
m0_CF   = model.m + 0 # Initial model of Epot
m0_DF   = np.zeros(n_DF) # Initial model of Eind
m0     = np.hstack((m0_CF, m0_DF)) # Mean of prior distribution

# Initial guess - Covariance
#C0_CF   = 1000**2 * np.diag(np.ones(n_CF)) # Covariance associated with m0_CF
C0_CF   = model.Cmpost + 0 # Covariance associated with m0_CF
C0_DF   = 500**2 * np.diag(np.ones(n_DF)) # Covariance associated with m0_DF
C0_CFDF = np.zeros((n_DF, n_CF)) # Covariance between m0_CF and m0_DF
C0      = np.vstack((np.hstack((C0_CF  , C0_CFDF.T)),
                     np.hstack((C0_CFDF, C0_DF    )) ))

#%% Step 3: Run Kalman filter

ms = np.zeros((n, nt))
rs = []
Cs = np.zeros((n, n, nt))

m_o = m0 # Old model (k-1)
C_o = C0 # Old covariance (k-1)

for i, t in tqdm(enumerate(times[:6]), total=nt):

    ## Prepatation
    # Define model function
    SH = SH_fun().reshape(-1, 1)
    SP = SP_fun().reshape(-1, 1)

    A = calc_dmf(SH, SP, C_o)

    # Define measurement function    
    model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    #model.add_data(amp_data, sm_data, sd_data)
    model.add_data(amp_data, sm_data)
    model.get_G_CF()
    model.get_G_DF()
    
    d = model._d
    nd = d.size
    #R = np.diag(1/model._w)
    R = model._Cd
    H = np.hstack((model.G_CF, model.G_DF))
    
    # Define process noise
    #Q = .1 * np.diag(np.diag(C_o))
    Q = .5 * C_o
    
    ## Prediction
    m_p = A @ m_o             # Prediction of current model (k)
    C_p = A @ C_o @ A.T + Q   # Prediction of current model covariance (k)
    
    ## Update
    # Calculate various things
    r_p         = d - H @ m_p                       # Calculate residual between data and predictions
    Cd_p        = H @ C_p @ H.T + np.diag(R)                 # Project predicted model covariance into data and add measurment noise
    Cd_p_inv    = np.linalg.pinv(Cd_p) # Inverse, needed below
    #Cd_p_inv    = 1/Cd_p # Inverse, needed below
    K           = C_p @ H.T @ Cd_p_inv              # Kalman gain
    
    # Actual update
    m_c = m_p + K @ r_p         # Update model (k)
    C_c = C_p - K @ Cd_p @ K.T  # Update model covariance (k)
        
    # Store filtered models
    ms[:, i] = m_c
    Cs[:, :, i] = C_c
    #rs = H @ m_c

#%%

fig, axs = plt.subplots(3,2, sharey=True)
axs = axs.flatten()
axs[0].plot(m0[:n_CF])
axr = plt.twinx(axs[0])
axr.plot(m0[n_CF:], color='tab:orange')
for j in range(1,6):
    axs[j].plot(ms[:, j-1][:n_CF])
    axr = plt.twinx(axs[j])
    axr.plot(ms[:, j-1][n_CF:], color='tab:orange')

#%%
plt.ioff()
apex = apexpy.Apex(2012, refh = 110)
savepath = path_out + 'figures/kalman_lompe/'

model.m = m0[:n_CF]
#model.m = (A @ m_o)[:n_CF]
savefile = savepath + 'init'
lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
plt.close('all')

for i, t in tqdm(enumerate(times[:6]), total=nt):
    model.m = ms[:n_CF, i]
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')
plt.ion()

#%% Step 4: Save filtered models

with open(path_out + '/data/KF_lompe_model_vectors.pkl', 'wb') as f:
    pickle.dump(ms, f)

#%% Step 5: Plots





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
