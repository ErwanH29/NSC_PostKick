import numpy as np
from scipy.interpolate import interp1d
from amuse.lab import units


### General Constants
AVG_STAR_MASS  = 1 | units.MSun
AVG_STAR_RAD   = 1 | units.RSun
TDE_FACTOR     = 0.9
SWITCH_FACTOR  = 1/3
DEPLETE_FACTOR = 0.75
H0  = 67.4 | (units.kms/units.Mpc)
OMEGA_M = 0.303
OMEGA_L = 0.697

### Press-Schechter function parameters -- Table A1: https://arxiv.org/pdf/1410.3485
z_bins   = np.array([0.05, 0.35, 0.75, 1.5, 2.5, 3.5, 4.0, 7.0])
phi_star = 10**-3 * np.array([0.84, 0.84, 0.74, 0.45, 0.22, 0.12, 0.12, 0.12])
M_norm   = 10**np.array([11.14, 11.11, 11.06, 10.91, 10.78, 10.60, 10.60, 10.60])
alpha_values    = np.array([-1.43, -1.45, -1.48, -1.57, -1.66, -1.74, -1.74, -1.74])
phi_star_interp = interp1d(z_bins, phi_star, kind='linear', fill_value='extrapolate')
M_star_interp   = interp1d(z_bins, M_norm, kind='linear', fill_value='extrapolate')
alpha_interp    = interp1d(z_bins, alpha_values, kind='linear', fill_value='extrapolate')

### Binary BH merger rates -- Fig. 10: https://arxiv.org/pdf/2412.15334
merger_rate_zbins = [0, 0.75, 1.25, 2, 2.6, 3.5, 4.0, 5.0, 6.0, 7.0]
merger_rate = [  
    [0.0006, 0.0007, 0.0037, 0.0020, 0.0055, 0.002,  0., 0., 0., 0.],  # IMBH-IMBH
    [0.0007, 0.0004, 0.0027, 0.0020, 0.0050, 0.,     0., 0., 0., 0.]   # SMBH-SMBH, 
]  # in yr^-1 Gpc^-3
merger_rate_IMBH = interp1d(
    merger_rate_zbins,
    merger_rate[0],
    kind='linear',
    fill_value='extrapolate'
)
merger_rate_SMBH = interp1d(
    merger_rate_zbins,
    merger_rate[1],
    kind='linear',
    fill_value='extrapolate'
)

### Probability Distributions -- Table VIII: https://arxiv.org/pdf/1201.1923
Prob_Distr = {
    "Kick Lower Limit": [    0,     100,      200,       300,      400,      500,     1000,     1500,    2000],
    "Hot Kick PDF":  [0.342593, 0.211364, 0.116901, 0.078400, 0.057590, 0.140283, 0.040183, 0.010309],
    "Cold Kick CDF": [0.414482, 0.283502, 0.125030, 0.070967, 0.042490, 0.059309, 0.004030, 0.000185]
}
vkick_bins      = np.array(Prob_Distr["Kick Lower Limit"], dtype=float)
Prob_Distr["Hot Kick PDF"] = np.array(Prob_Distr["Hot Kick PDF"], dtype=float)
Prob_Distr["Hot Kick PDF"] /= Prob_Distr["Hot Kick PDF"].sum()