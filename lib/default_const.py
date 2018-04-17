#!/usr/bin/env python3
# UTF-8, *nix

from numpy import array


### parameters of the linear trend
alpha = 0.1     
beta  = 0.05   

### SNR (signal-to-noise)
gamma = 0.50   

### harmonic component
A1    = 1.0  # amplitude
nu_1  = 0.1  # frequency (Hz)    
phi_1 = 0.0  # phase

### time step
dt    = 1.0 # (sec) 

### size of series
N     = 230

### probability of the peak
q     = 0.01

### critical noise value
X1    = 9.0


nodes = array([dt*k for k in range(N)])
