#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: grabuffo
"""

import os
import math
import pickle
import gzip
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.cm as cm


from tvb.basic.neotraits.api import HasTraits, NArray, Attr, Range, Final, List
from tvb.simulator.coupling import SparseCoupling
from tvb.simulator.coupling import Coupling
from tvb.simulator.models.base import Model
from tvb.simulator.lab import *
import math

# from tvb.basic.neotraits.api import HasTraits, NArray, Attr, Range, Final, List
# from tvb.simulator.coupling import SparseCoupling
# from tvb.simulator.coupling import Coupling
# from tvb.simulator.models.base import Model
# from tvb.simulator.lab import *

##########################################

# Network information

with open('../data/Allen_148/region_labels.txt') as f:
    content = f.readlines()  
ROIs = [ix.strip() for ix in content] # ROIs names

nregions=len(ROIs)

# Artifactual ROIs
remove_roi=[27, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 
                101, 129, 130,  131,  132,  133,  134,  135,  136,  137,  138,  139,  140, 141, 142, 143,144,145,146,147,
                16, 19, 21, 23, 24,
                16+74, 19+74, 21+74, 23+74, 24+74]

# Non-artefactual ROIs
reroi=np.delete(np.arange(nregions),remove_roi)

Cortical_labels=np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86, 87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103])

# Resting State Network classification
RSNs={
    'DMN': [15, 16, 17, 19, 7, 4, 64, 65, 66, 68, 56, 53], 
    'Vis': [11, 14, 10, 12, 13, 9, 60, 63, 59, 61, 62, 58], 
    'LCN': [1, 2, 3, 5, 6, 8, 18, 20, 21, 50, 51, 52, 54, 55, 57, 67, 69, 70], 
    'BF': [22, 23, 30, 31, 32, 33, 34, 35, 36, 37, 47, 48, 0, 71, 72, 79, 80, 81, 82, 83, 84, 85, 86, 96, 97, 49], 
    'HPF': [24, 25, 26, 27, 28, 29, 73, 74, 75, 76, 77, 78], 
    'Th': [38, 39, 40, 41, 42, 43, 44, 45, 46, 87, 88, 89, 90, 91, 92, 93, 94, 95]
}

RSNs_mod=list(RSNs.keys())
short_mod=RSNs.keys()
lenmods=[len(RSNs['DMN']),len(RSNs['Vis']),len(RSNs['LCN']),len(RSNs['BF']),len(RSNs['HPF']),len(RSNs['Th'])]
order_mod=np.asarray(RSNs['DMN']+RSNs['Vis']+RSNs['LCN']+RSNs['BF']+RSNs['HPF']+RSNs['Th'])
rex = np.ix_(order_mod,order_mod)


##########################################


# Define Buendia Model 
class MF_LG_CouplingR(Model):
    #coupling_variables = {'coupling': np.r_[0]} # this is needed for configuration
    @property
    def cvar(self):
        self.log.warning("Use cvars defined in the coupling_variables instead")
    # define number of state variables and their names
    _nvar = 2
    state_variables = ('R', 'psi')
    # expose state variables for the monitors
    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("R", "psi"),
        default=("R", "psi"))
    # for each state variable define range of values for initial conditions
    state_variable_range = Final(
        default={"R": np.array([0.,1.]),
                 "psi": np.array([0.,2*math.pi])})
    # for each state variable define bonudaries
    state_variable_boundaries = Final(
        label="Boundary for R",
        default={"R": np.array([0.000001, np.infty])})
    # indices of variables entering the node-to-node coupling
    cvar = np.array([0,1],dtype=np.int32)
    #cvar = np.array([0,1],dtype=np.int32)
    # parameters can be defined like this
    J     = NArray(default=np.array([1.]),)
    sigma = NArray(default=np.array([0.6]),)
    a     = NArray(default=np.array([1.25]),)
    omega = NArray(default=np.array([1.]),)

    def dfun(self, state_variables, coupling, local_coupling):
        # to make the equations more readable
        R, psi = state_variables
        J      = self.J
        a      = self.a
        sigma  = self.sigma
        omega  = self.omega
 
        coupl = coupling[0,:] # this refers to a special KuramotoR coupling defined by me

        derivative = np.empty_like(state_variables)

        # fill in the equations here
        derivative[0] = 0.5*R*(J*(1-R**2)-sigma**2) - 0.5*a*(1-R**2)*np.cos(psi)
        derivative[1] = omega + (a*(1+R**2)*np.sin(psi))/(2*R) + coupl 
        
        return derivative

# Define coupling function
class KuramotoR(SparseCoupling):
    r"""
    Provides a Kuramoto-style coupling, a periodic difference of the form
    
    .. math::
        a G_ij y_j sin(x_j - x_i)
    
    """
    a = NArray(
        label=":math:`a`",
        default=np.array([1.0,]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="Rescales the connection strength.",)

    def __str__(self):
        return simple_gen_astr(self, 'a')

    def pre(self, x_i, x_j):
        return x_j*np.sin(x_j[::-1,:,:] - x_i[::-1,:,:])

    def post(self, gx):
        return self.a * gx


# Define simulator function
def simulate_network_conduct_seed_wholeR(Allen_SC,J,a,sigma,omega, g,eta, ini, v, sim_length,dt,verbose=False):
    # Initial conditions
    r0 = np.random.uniform(low=0.01,high=1.,size=((1,1,Allen_SC.number_of_regions,1)))
    psi0 = np.random.uniform(low=0.,high=2*math.pi,size=((1,1,Allen_SC.number_of_regions,1)))
    init_cond = np.concatenate([r0,psi0],axis=1)
    # Noise
    hiss = noise.Additive(nsig=np.array([0,eta]),noise_seed=ini)
    #Configure Simulator
    sim = simulator.Simulator(
        connectivity=Allen_SC,
        model= MF_LG_CouplingR(J=np.array([J]),a=np.array([a]),sigma=np.array([sigma]),omega=np.array([omega])),
        integrator=integrators.HeunStochastic(noise=hiss,dt=dt),
        coupling=KuramotoR(a=np.r_[g]),
        conduction_speed=v,
        monitors=[monitors.Raw(),monitors.AfferentCoupling()]
    )
    sim.initial_conditions = init_cond
    sim.configure()
    # Run simulation
    ((time, data), (_,coupl_monitor)) = sim.run(simulation_length=sim_length)

    # Simulated time series
    r = data[:,0,:,0] # R~(time,ROIs)
    psi = data[:,1,:,0] # PSI~(time,ROIs)
    afferent_inputs = coupl_monitor[:,0,:,0] # NETWORK_INPUTS~(time,ROIs)
    
    return time, r, psi, afferent_inputs

def simulate_network_conduct_seed_wholeBOLD_R(Allen_SC,J,a,sigma,omega, g,eta, ini, v, sim_length,dt,verbose=False):
    # Initial conditions
    r0 = np.random.uniform(low=0.01,high=1.,size=((1,1,Allen_SC.number_of_regions,1)))
    psi0 = np.random.uniform(low=0.,high=2*math.pi,size=((1,1,Allen_SC.number_of_regions,1)))
    init_cond = np.concatenate([r0,psi0],axis=1)
    # Noise
    hiss = noise.Additive(nsig=np.array([0,eta]),noise_seed=ini)
    #Configure Simulator
    sim = simulator.Simulator(
        connectivity=Allen_SC,
        model= MF_LG_CouplingR(J=np.array([J]),a=np.array([a]),sigma=np.array([sigma]),omega=np.array([omega])),
        integrator=integrators.HeunStochastic(noise=hiss,dt=dt),
        coupling=KuramotoR(a=np.r_[g]),
        conduction_speed=v,
        monitors=[monitors.Bold(period=1000),monitors.Raw(),monitors.AfferentCoupling()]
    )
    sim.initial_conditions = init_cond
    sim.configure()
    # Run simulation
    ((time_b, data_b), (time, data), (_,coupl_monitor)) = sim.run(simulation_length=sim_length)

    # Simulated time series
    Br = data_b[:,0,:,0] # R~(time,ROIs)
    Bpsi = data_b[:,1,:,0] # PSI~(time,ROIs)

    r = data[:,0,:,0]
    psi = data[:,1,:,0]
    
    if r[r>1].shape[0] != 0:
        verbose and print('Warning: R exceeds 1 for g = '+str(round(g,5))+', η = '+str(round(eta,5)))
        return time, r, psi, traces
    
    return time_b, Br, Bpsi, time, r, psi, coupl_monitor[:,0,:,0]

# def simulate_network_conduct_seed_wholeBOLD(Allen_SC,J,a,sigma,omega, g,eta, ini, v, sim_length,dt,verbose=False):
#     # Initial conditions
#     r0 = np.random.uniform(low=0.01,high=1.,size=((1,1,Allen_SC.number_of_regions,1)))
#     psi0 = np.random.uniform(low=0.,high=2*math.pi,size=((1,1,Allen_SC.number_of_regions,1)))
#     init_cond = np.concatenate([r0,psi0],axis=1)
#     # Noise
#     hiss = noise.Additive(nsig=np.array([0,eta]),noise_seed=ini)
#     #Configure Simulator
#     sim = simulator.Simulator(
#         connectivity=Allen_SC,
#         model= MF_LG_CouplingR(J=np.array([J]),a=np.array([a]),sigma=np.array([sigma]),omega=np.array([omega])),
#         integrator=integrators.HeunStochastic(noise=hiss,dt=dt),
#         coupling=KuramotoR(a=np.r_[g]),
#         conduction_speed=v,
#         monitors=[monitors.Bold(period=1000)]
#     )
#     sim.initial_conditions = init_cond
#     sim.configure()
#     # Run simulation
#     (time, data), = sim.run(simulation_length=sim_length)

#     # Simulated time series
#     Br = data[:,0,:,0] # R~(time,ROIs)
#     Bpsi = data[:,1,:,0] # PSI~(time,ROIs)
    
#     return time, Br, Bpsi

def filter_simulated_rois(simulated_bold):
    # simulated_bold ~ (time, 148)
    # since the Allen parcellation with 148 regions used in the brain simulations contains some parts of brain regions (e.g., ACAl and ACAv) 
    # that are not present in the Grandjean parcellation (e.g., we only have the entire brain region ACA), 
    # we first define the simulated activity in these regions as their average (e.g., ACAl(t)=ACAv(t)= (ACAv(t)+ACAl(t))/2) 
    # and then we remove the duplicate regions. 
    # Furthermore, we eliminate other brain regions that contain artifacts in the data.
    # This way, we can compare simulated and empirical data considering 98 ROIs
    nreg=int(simulated_bold.shape[1]/2)
    region_groups = [(19, 20), (17, 18), (14, 15), (21, 22, 23), 
                     (19+nreg, 20+nreg), (17+nreg, 18+nreg), (14+nreg, 15+nreg), (21+nreg, 22+nreg, 23+nreg)]  
    # groups of ROIs in Allen148 that correspond to the same Grandjean ROI
    # Redefine the data
    for group in region_groups:
        # Calculate the mean across specified regions
        mean_activity = simulated_bold[:, group].mean(axis=1, keepdims=True)
        # Assign the mean back to each region in the group
        simulated_bold[:, group] = mean_activity
    
    simulated_bold_reduced=simulated_bold[:,:]#[:,reroi]
    
    return simulated_bold_reduced

def set_up_connectivity148(file_name, v):
    A148_con = connectivity.Connectivity.from_file(file_name)
    np.fill_diagonal(A148_con.weights, 0.)
    A148_con.weights = A148_con.weights/np.max(A148_con.weights) #normalization
    A148_SC = A148_con.weights
    # Setup the connectivity for the simulator
    Allen_SC = connectivity.Connectivity(
        weights= A148_SC,
        tract_lengths=A148_con.tract_lengths,
        speed= np.asarray(v),
        centres = A148_con.centres,
        region_labels=np.asarray([i for i in range(len(A148_SC))],dtype='<U128')) 
    Allen_SC.configure()
    return Allen_SC

# # Define the retry function
# def retry_function(func, attempts=5, delay=1):
#     for _ in range(attempts):
#         try:
#             return func()
#         except Exception as e:
#             print(f"Attempt failed: {e}")
#             time.sleep(delay)
#     # If all attempts fail, return None
#     return None