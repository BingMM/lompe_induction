#%% Import

import os
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
    sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', error = 200, iweight=1.0)
    
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

times = pd.date_range('2012-04-05 00:00', '2012-04-05 23:59', freq = '3Min')[:100] # Limiting testing to the first 100, instead of all 480 times
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

A_CFDF = np.zeros((n_DF, n_CF))
A_DFDF = np.eye(n_DF)

A_DF = np.hstack((A_CFDF, A_DFDF))

#%% Step 1: Prepare Dynamic model function (CF)

A_CFCF = np.eye(n_CF)

# Br from CF E
## Include temporal shift
#lon_shift = grid.lon.flatten() + (360/24/60*3)
#lon_shift[lon_shift>180] -= 360
lon_shift = grid.lon_mesh.flatten() + (360/24/60*3)
lon_shift[lon_shift>180] -= 360
xi_shift, eta_shift = grid.projection.geo2cube(lon_shift.reshape(grid.xi_mesh.shape), grid.lat_mesh)

Q_shift = np.zeros((grid.xi_mesh.size, grid.xi_mesh.size))

for row in range(grid.xi_mesh.shape[0]):
    for col in range(grid.xi_mesh.shape[1]):
        row_i = np.argmin(abs(grid.eta_mesh[:, 0]-eta_shift[row, col]))
        col_i = np.argmin(abs(grid.xi_mesh[0, :]-xi_shift[row, col]))
        
        Q_shift[row*grid.xi_mesh.shape[1]+col, row_i*grid.xi_mesh.shape[1]+col_i] = 1

#A = np.random.normal(0, 10, (grid.xi_mesh.shape))
#A = np.arange(grid.xi_mesh.size).reshape(grid.xi_mesh.shape)
#A_prime = (Q_shift @ A.flatten()).reshape(grid.xi_mesh.shape)

#vmax = np.max(abs(np.vstack((A, A_prime))))
#clvls=np.linspace(-vmax, vmax, 40)
#fig, axs = plt.subplots(1, 2)
#axs[0].tricontourf(grid.xi_mesh.flatten(), grid.eta_mesh.flatten(), A.flatten(), levels=clvls, cmap='bwr')
#axs[1].tricontourf(xi_shift.flatten(), eta_shift.flatten(), A_prime.flatten(), levels=clvls, cmap='bwr')
#axs[0].imshow(A)
#axs[1].imshow(A_prime)

## DF SECS design matrix for Br at ionosphere
_, _, Hu = get_SECS_B_G_matrices(grid.lat_mesh.flatten(), grid.lon_mesh.flatten(), model.R, 
                                 grid.lat.flatten(), grid.lon.flatten(),
                                 current_type = 'divergence_free',
                                 RI = model.R,
                                 singularity_limit = model.secs_singularity_limit)

_, _, Hu_t = get_SECS_B_G_matrices(grid.lat_mesh.flatten(), lon_shift, model.R, 
                                   grid.lat.flatten(), grid.lon.flatten(),
                                   current_type = 'divergence_free',
                                   RI = model.R,
                                   singularity_limit = model.secs_singularity_limit)

## QA 
QiA = model.QiA

## HQiA
HQiA_u   = Hu @ QiA
HQiA_u_t = Hu_t @ QiA

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

def calc_G_r_CF(SH, SP, t_shift=False):
    c_J_CF = - Dn.dot(SP) * Ee + De.dot(SP) * En \
             - Dn.dot(SH) * En * hemisphere - De.dot(SH) * Ee * hemisphere \
             - SH * DdivdotE * hemisphere
    if t_shift:
        G_r_CF = HQiA_u_t.dot(c_J_CF)
    else:
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
    T = c_E @ C_i[n_CF:, n_CF:] @ c_E.T
    return T

def calc_A_CFDF(SH, SP, C_i, l1=0, l2=0):
    G_r_CF  = calc_G_r_CF(SH, SP)
    T       = calc_T(G_r_CF, C_i)
    #T       = np.eye(T.shape[0])
    T_inv   = scipy.linalg.lstsq(T, np.eye(T.shape[0]), lapack_driver='gelsy')[0]
    del T
    
    GTT     = G_r_CF.T @ T_inv
    del T_inv

    GTTG    = GTT @ G_r_CF
    del G_r_CF
    
    GTTcE   = GTT @ c_E
    del GTT
    
    gtgmag = np.median(np.diag(GTTG))
    #A_CFDF = np.linalg.lstsq(GTTG + 1e3*gtgmag*np.eye(GTTG.shape[0]), GTTcE, rcond=None)[0]
    #A_CFDF = np.linalg.solve(GTTG + 1e4*gtgmag*np.eye(GTTG.shape[0]), GTTcE)
    
    #LTL = model.Le.T.dot(model.Le)
    #ltlmag = np.median(np.diag(LTL))
    #A_CFDF = np.linalg.lstsq(GTTG + l1*gtgmag*np.eye(GTTG.shape[0]) + l2*gtgmag/ltlmag*LTL, GTTcE, rcond=None)[0]
    A_CFDF = scipy.linalg.lstsq(GTTG + gtgmag*LTL_A_CFDF, GTTcE, lapack_driver='gelsy')[0]
    
    del GTTG, GTTcE
    
    return A_CFDF

'''
def calc_A_CF(SH, SP, C_i, l1=0, l2=0):
    G_r_CF = calc_G_r_CF(SH, SP)
    
    G_r_CF_t = calc_G_r_CF(SH, SP, t_shift=False)
        
    T = G_r_CF_t @ C_i[:n_CF, :n_CF] @ G_r_CF_t.T + c_E @ C_i[n_CF:, n_CF:] @ c_E.T
    T_inv = scipy.linalg.lstsq(T, np.eye(T.shape[0]), lapack_driver='gelsy')[0]
    del T
    
    GTT = G_r_CF.T @ T_inv
    del T_inv
    
    GTTG = GTT @ G_r_CF
    del G_r_CF
    
    GTTG_t = GTT @ G_r_CF_t
    del G_r_CF_t
    
    GTTcE   = GTT @ c_E
    del GTT
    
    gtgmag = np.median(np.diag(GTTG))
        
    A_CF = scipy.linalg.lstsq(GTTG + gtgmag*LTL_A_CFDF, np.hstack((GTTG_t, GTTcE)), lapack_driver='gelsy')[0]
    del GTTG, GTTG_t, GTTcE
    
    return A_CF
'''

def calc_A_CF(SH, SP, C_i, l1=0, l2=0):
    G_r_CF = calc_G_r_CF(SH, SP)
        
    
    G_r_CF_t = calc_G_r_CF(SH, SP, t_shift=False)    
    
    T = G_r_CF_t @ C_i[:n_CF, :n_CF] @ G_r_CF_t.T
    T_inv = scipy.linalg.lstsq(T, np.eye(T.shape[0]), lapack_driver='gelsy')[0]
    del T
    
    GTT = G_r_CF.T @ T_inv
    del T_inv
    
    GTTG = GTT @ G_r_CF
    gtgmag = np.median(np.diag(GTTG))

    GTTG_t = GTT @ G_r_CF_t
    del G_r_CF_t, GTT

    #A_CFCF = scipy.linalg.lstsq(GTTG + gtgmag*LTL_A_CFDF, GTTG_t, lapack_driver='gelsy')[0]
    #A_CFCF = scipy.linalg.lstsq(GTTG, GTTG_t, lapack_driver='gelsy')[0]
    A_CFCF = scipy.linalg.lstsq(GTTG + gtgmag*LTL_A_CFCF, GTTG_t, lapack_driver='gelsy')[0]
    #A_CFCF = np.eye(n_CF)
    del GTTG, GTTG_t
    
    '''
    A_CFCF = Q_shift
    '''

    T = c_E @ C_i[n_CF:, n_CF:] @ c_E.T
    T_inv = scipy.linalg.lstsq(T, np.eye(T.shape[0]), lapack_driver='gelsy')[0]
    del T
    
    GTT = G_r_CF.T @ T_inv
    del T_inv
    
    GTTG = GTT @ G_r_CF
    gtgmag = np.median(np.diag(GTTG))
    
    #GTTcE   = GTT @ c_E
    GTTcE   = GTT @ (Q_shift @ c_E)
    del GTT
    
    A_CFDF = scipy.linalg.lstsq(GTTG + gtgmag*LTL_A_CFDF, GTTcE, lapack_driver='gelsy')[0]    
    del GTTG, GTTcE
    
    
    return np.hstack((A_CFCF, A_CFDF))



#%% Step 1: Prepare Dynamic model function

def calc_dmf(SH, SP, C_i, l1=0, l2=0):
    #A_CFDF = calc_A_CFDF(SH, SP, C_i, l1, l2)
    #A_CF = np.hstack((A_CFCF, A_CFDF))
    A_CF = calc_A_CF(SH, SP, C_i, l1, l2)
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
C0_DFCF = np.zeros((n_DF, n_CF)) # Covariance between m0_CF and m0_DF
C0_CFDF = np.zeros((n_CF, n_DF))
C0      = np.vstack((np.hstack((C0_CF  , C0_CFDF)),
                     np.hstack((C0_DFCF, C0_DF    )) ))

#%% Step 3: Run Kalman filter

ms = np.zeros((n, nt))
Cs = np.zeros((n, n, nt))
rs = []
rmses = np.zeros(nt)
n_effs = np.zeros(nt)

m_o = m0 # Old model (k-1)
C_o = C0 # Old covariance (k-1)

l1 = 1e3
l2 = 1e0

LTL_CFCF = model.Le.T.dot(model.Le)
ltlmag = np.median(np.diag(LTL_CFCF))
LTL = np.vstack((np.hstack((LTL_CFCF,               np.zeros((n_CF, n_DF)))),
                 np.hstack((np.zeros((n_DF, n_CF)), np.zeros((n_DF, n_DF)))) ))
LTL = 1e-1*np.eye(LTL.shape[0]) + 1e-1*LTL/ltlmag

LTL_A_CFDF = l1*np.eye(LTL_CFCF.shape[0]) + l2*LTL_CFCF/ltlmag
#LTL_A_CFCF = 1e-1*np.eye(LTL_CFCF.shape[0]) + 1e-1*LTL_CFCF/ltlmag
LTL_A_CFCF = 1e-1*np.eye(LTL_CFCF.shape[0])

# Save for smoother
As = []

for i, t in tqdm(enumerate(times), total=nt):

    ## Prepatation
    # Define model function
    SH = SH_fun().reshape(-1, 1)
    SP = SP_fun().reshape(-1, 1)

    A = calc_dmf(SH, SP, C_o, l1=l1, l2=l2)

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
    R = 1/model._w
    #R = model._Cd
    H = np.hstack((model.G_CF, model.G_DF))
    
    # Define process noise
    #Q = .1 * np.diag(np.diag(C_o))
    #Q = .1 * C_o
    #Q = 0.01 * C_o
    #Q = 0 * C_o
    #Q = 0.1 * C_o
    #Q = 0.1 * C_o
    Q = .05 * np.diag(np.diag(C0))
    '''
    C0_scale = np.max(np.diag(C0))
    Q = np.diag(np.diag(C_o)) / np.max(np.diag(C_o)) * C0_scale
    '''
    #Q = .05 * C_o
    #Q = .05 * np.diag(np.diag(C_o))
    
    ## Prediction
    m_p = A @ m_o             # Prediction of current model (k)
    #C_p = A @ C_o @ A.T + Q   # Prediction of current model covariance (k)
    
    C_p_inv = scipy.linalg.lstsq(A @ C_o @ A.T, np.eye(A.shape[0]), lapack_driver='gelsy')[0]
    #C_p_inv = np.linalg.pinv(A @ C_o @ A.T)
    Cpmag = np.median(np.diag(C_p_inv))
    #C_p = np.linalg.pinv(C_p_inv + Cpmag*LTL) + Q
    C_p = scipy.linalg.lstsq(C_p_inv + Cpmag*LTL, np.eye(C_p_inv.shape[0]), lapack_driver='gelsy')[0] + Q
    '''
    C_p = scipy.linalg.lstsq(C_p_inv + Cpmag*LTL, np.eye(C_p_inv.shape[0]), lapack_driver='gelsy')[0]
    
    C_p_scale = np.max(np.diag(C_p))
    if C_p_scale < C0_scale:
        print('Variance decreasing, scaling')
        Q = Q * (abs(C_p_scale - C0_scale) / C0_scale)
        C_p += Q
    '''
    
    ## Update
    # Calculate various things
    r_p         = d - H @ m_p                       # Calculate residual between data and predictions
    CpHT        = C_p @ H.T
    Cd_p        = H @ CpHT              # Project predicted model covariance into data and add measurment noise
    tmp         = np.einsum('ii->i', Cd_p)
    tmp         += R
    #Cd_p        = H @ CpHT + np.diag(R)                 # Project predicted model covariance into data and add measurment noise
    
    Cd_p_inv    = scipy.linalg.lstsq(Cd_p, np.eye(Cd_p.shape[0]), lapack_driver='gelsy')[0]
    #Cd_p_inv    = np.linalg.pinv(Cd_p) # Inverse, needed below
    K           = CpHT @ Cd_p_inv              # Kalman gain
    
    # Actual update
    m_c = m_p + K @ r_p         # Update model (k)
    C_c = C_p - K @ Cd_p @ K.T  # Update model covariance (k)
        
    # Store filtered models
    ms[:, i] = m_c
    Cs[:, :, i] = C_c
    rs.append(d - H @ m_c)
    
    #rmses[i] = rs[i].T @ np.diag(1/R) @ rs[i]
    rmses[i] = (rs[i] / R).T @ rs[i]
    
    HTR = H.T * (1/R)
    Rd = H @ scipy.linalg.lstsq(HTR @ H, np.eye(HTR.shape[0]), lapack_driver='gelsy')[0] @ HTR
    #Rd = H @ np.linalg.pinv(HTR @ H) @ HTR
    n_effs[i] = np.trace(Rd)
    
    m_o = m_c
    C_o = C_c
    
    As.append(A)

#%%
'''
with open('/home/bing/Dropbox/work/temp_storage/As.pkl', 'wb') as f:
    pickle.dump(As, f)

with open('/home/bing/Dropbox/work/temp_storage/ms.pkl', 'wb') as f:
    pickle.dump(ms, f)

with open('/home/bing/Dropbox/work/temp_storage/Cs.pkl', 'wb') as f:
    pickle.dump(Cs, f)
'''
#%%
'''
with open('/home/bing/Dropbox/work/temp_storage/As.pkl', 'rb') as f:
    As = pickle.load(f)

with open('/home/bing/Dropbox/work/temp_storage/ms.pkl', 'rb') as f:
    ms = pickle.load(f)

with open('/home/bing/Dropbox/work/temp_storage/Cs.pkl', 'rb') as f:
    Cs = pickle.load(f)
'''
#%% Smoother

ms_smooth = np.zeros((n, nt))
Cs_smooth = np.zeros((n, n, nt))

for i in tqdm(range(nt-1, 0-1, -1), total=nt):
    if i == 99:
        ms_smooth[:, i] = ms[:, i]
        Cs_smooth[:, :, i] = Cs[:, :, i]
    else:
        A = As[i]
        m = ms[:, i]
        C = Cs[:, :, i]
    
        m_guess = A.dot(m)
        C_guess = A.dot(C).dot(A.T) + Q
    
        Ck = C.dot(A.T).dot(scipy.linalg.lstsq(C_guess, np.eye(C_guess.shape[0]), lapack_driver='gelsy')[0])
        ms_smooth[:, i] = m + Ck.dot(ms_smooth[:, i+1] - m_guess)
        Cs_smooth[:, :, i] = C + Ck.dot(Cs_smooth[:, :, i+1] - C_guess).dot(Ck.T)
        
#%%

chi_l = scipy.stats.chi2.ppf(.025, n_effs)
chi_u = scipy.stats.chi2.ppf(.975, n_effs)

plt.figure()
plt.fill_between(np.arange(nt), chi_l, chi_u, color='tab:blue', alpha=.4, label='95% confidence')
plt.plot(chi_l, '--', color='k')
plt.plot(chi_u, '--', color='k')
plt.plot(n_effs, color='k', label='degrees of freedom')
plt.plot(rmses, color='tab:orange', label='Chi squared')

plt.xlabel('Time step')
plt.ylabel('Chi squared')
plt.title('0.1*C0 : l1=1e-1 : l2=1e-1 : With spatial weights')


#%%

Eind_max = np.max(abs(ms[n_CF:, :]))

fig, axs = plt.subplots(3,2, sharey=True)
axs = axs.flatten()
axs[0].plot(m0[:n_CF])
axr = plt.twinx(axs[0])
axr.plot(m0[n_CF:], color='tab:orange')
axr.set_ylim([-Eind_max, Eind_max])
for j in range(1,6):
    axs[j].plot(ms[:, j-1][:n_CF])
    axr = plt.twinx(axs[j])
    axr.plot(ms[:, j-1][n_CF:], color='tab:orange')
    axr.set_ylim([-Eind_max, Eind_max])

#%% Plot m_CF

plt.ioff()
apex = apexpy.Apex(2012, refh = 110)
#savepath = path_out + 'figures/kalman_m_CF_1e{}_1e{}/'.format(int(np.log10(l1)), int(np.log10(l2)))
savepath = path_out + 'figures/kalman_m_CF_test_new/'
try:
    os.mkdir(savepath)
except:
    print('Folder already exists')

t = times[0]
model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
model.add_data(amp_data, sm_data, sd_data)
model.m = m0[:n_CF]
#model.m = (A @ m_o)[:n_CF]
savefile = savepath + 'init'
lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
plt.close('all')

for i, t in tqdm(enumerate(times), total=nt):
    model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    model.add_data(amp_data, sm_data, sd_data) 
    model.m = ms[:n_CF, i]
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')
plt.ion()

#%% Plot m_CF smoothed

plt.ioff()
apex = apexpy.Apex(2012, refh = 110)
savepath = path_out + 'figures/kalman_m_CF_test_smoothed/'
try:
    os.mkdir(savepath)
except:
    print('Folder already exists')

t = times[0]
model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
model.add_data(amp_data, sm_data, sd_data)
model.m = m0[:n_CF]
#model.m = (A @ m_o)[:n_CF]
savefile = savepath + 'init'
lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
plt.close('all')

for i, t in tqdm(enumerate(times), total=nt):
    model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    model.add_data(amp_data, sm_data, sd_data) 
    model.m = ms_smooth[:n_CF, i]
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')
plt.ion()

#%% Plot m_DF

# Default arrow scales (all SI units):
QUIVERSCALES = {'ground_mag':       600 * 1e-9 , # ground magnetic field scale [T]
                'space_mag_fac':    600 * 1e-9 , # FAC magnetic field scale [T]
                'convection':       2000       , # convection velocity scale [m/s]
                'efield':           100  * 1e-3, # electric field scale [V/m]
                'electric_current': 200 * 1e-3, # electric surface current density [A/m] Ohm's law 
                'secs_current':     1000 * 1e-3, # electric surface current density [A/m] SECS 
                'space_mag_full':   600 * 1e-9 } # FAC magnetic field scale [T]

# Default color scales (SI units):
COLORSCALES =  {'fac':        np.linspace(-.5, .5, 40) * 1e-6 * 2,
                'ground_mag': np.linspace(-50, 50, 50) * 1e-9, # upward component
                'hall':       np.linspace(0, 20, 32), # mho
                'pedersen':   np.linspace(0, 20, 32)} # mho

plt.ioff()
apex = apexpy.Apex(2012, refh = 110)
#savepath = path_out + 'figures/kalman_m_DF_1e{}_1e{}/'.format(int(np.log10(l1)), int(np.log10(l2)))
savepath = path_out + 'figures/kalman_m_DF_test_new/'
try:
    os.mkdir(savepath)
except:
    print('Folder already exists')

t = times[0]
model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
model.add_data(amp_data, sm_data, sd_data)
model.m = m0[n_CF:]
#model.m = (A @ m_o)[n_CF:]
savefile = savepath + 'init'
lompe.lompeplot_DF(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200}, 
                   quiverscales=QUIVERSCALES, colorscales=COLORSCALES)
plt.close('all')

for i, t in tqdm(enumerate(times), total=nt):
    model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    model.add_data(amp_data, sm_data, sd_data)
    model.m = ms[n_CF:, i]
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot_DF(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200}, 
                       quiverscales=QUIVERSCALES, colorscales=COLORSCALES)
    plt.close('all')
plt.ion()

#%% Plot m_DF

plt.ioff()
apex = apexpy.Apex(2012, refh = 110)
#savepath = path_out + 'figures/kalman_m_DF_1e{}_1e{}/'.format(int(np.log10(l1)), int(np.log10(l2)))
savepath = path_out + 'figures/kalman_m_DF_test_smooth/'
try:
    os.mkdir(savepath)
except:
    print('Folder already exists')

t = times[0]
model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
model.add_data(amp_data, sm_data, sd_data)
model.m = m0[n_CF:]
#model.m = (A @ m_o)[n_CF:]
savefile = savepath + 'init'
lompe.lompeplot_DF(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200}, 
                   quiverscales=QUIVERSCALES, colorscales=COLORSCALES)
plt.close('all')

for i, t in tqdm(enumerate(times), total=nt):
    model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    model.add_data(amp_data, sm_data, sd_data)
    model.m = ms_smooth[n_CF:, i]
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot_DF(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200}, 
                       quiverscales=QUIVERSCALES, colorscales=COLORSCALES)
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
