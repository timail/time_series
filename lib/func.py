# UTF-8, *nix
from random import normalvariate as random

from numpy import (
    array, 
    cos, sqrt,
    pi, 
)

import lib.default_const as data


t = lambda k: data.dt*k

X1 = data.X1

def gen_series(
    ### parameters of the linear trend
        alpha = data.alpha, 
        beta  = data.beta,

    ### SNR (signal-to-noise)
        gamma = data.gamma, 

    ### harmonic element
        A1    = data.A1,     # amplitude
        nu_1  = data.nu_1,   # frequency (Hz)
        phi_1 = data.phi_1,  # phase

    ### time step   
        dt = data.dt, # (sec)

    ### size of series    
        N = data.N,

    ### probability of the peak   
        q = data.q,
):

    sigma = sqrt(A1**2 / (2*gamma))

    noise = lambda: random(
        mu = 0,
        sigma = sigma,
    )


    return array(
        [   
            # linear trend
            alpha + beta * t + 
            # harmonic component
            A1 * cos(2 * pi * nu_1 * t - phi_1) + 
            # noise component
            sigma * noise()

            for t in data.nodes
        ]
    )
