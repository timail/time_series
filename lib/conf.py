# UTF-8, *nix

import lib.default_const as data


### Tukey window parameters
a_Tukey = 0.25
N_ratio = 0.50 # N* = ratio*N

N_Tukey = int(N_ratio * data.N)
