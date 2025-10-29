import numpy as np
from scipy import integrate as integ
from scipy.interpolate import interp1d

from amuse.units import units, constants

from plot.forecasting.forecast_parameters import H0, OMEGA_M, OMEGA_L

def get_Ez(z):
    return np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_L)

def get_look_back(z):
    """
    Compute look-back time in years.
    Args:
        z (float): Redshift.
    Returns: Look-back time in units.time.
    """
    integrand = lambda zp: (1.0 / ((1.0 + zp) * get_Ez(zp)))
    val = integ.quad(integrand, 0., z)[0]  # dimensionless
    return val/H0

def get_cosmic_time(z):
    """
    Get age of the Universe at redshift z in years.
    Args:
        z (float): Redshift.
    Returns: Cosmic time in units.time.
    """
    t0 = get_look_back(np.inf)
    return (t0 - get_look_back(z))

def get_comoving_distance(z):
    """
    Compute comoving distance in Mpc.
    Args:
        z (float): Redshift.
    Returns: Comoving distance in units.length.
    """
    z = np.atleast_1d(z)
    integrand = lambda zp: 1.0 / get_Ez(zp)
    Dc_vals = np.array([
        integ.quad(integrand, 0.0, zi)[0] for zi in z
    ])
    return (constants.c / H0) * Dc_vals

def get_dV_dz(z):
    """
    Compute the comoving volume element dV/dz in Mpc^3.
    Args:
        z (float): Redshift.
    Returns: Comoving volume element in units.length**3.
    """
    Dc = get_comoving_distance(z)
    dV_dz = (4 * np.pi) * (constants.c / H0) * Dc**2 / get_Ez(z)
    return dV_dz

def get_dt_dz(z):
    """
    Compute dt/dz in years.
    Args:
        z (float): Redshift.
    Returns: dt/dz in units.time.
    """
    Hz = H0 * get_Ez(z)
    dt_dz = 1.0 / ((1.0 + z) * Hz)
    return (dt_dz.in_(units.yr))

def R_gg(z, Rm_interp):
    """
    BH-BH merger rate in [yr^-1] from arXiv:2412.15334.
    Args:
        z (float): Redshift.
        Rm_interp (function): Interpolator for the merger rate.
    Returns: Merger rate in units.time
    """
    return Rm_interp(z) | units.yr**-1 * units.Gpc**-3