import numpy as np

from amuse.units import units, constants

def get_vdisp(mBH):
    """
    Get the velocity dispersion for a galaxy hosting a black hole of mass mBH.
    Uses Ferrares & Ford 2005 relation
    Args:
        mBH (units.mass): Mass of the black hole in solar masses.
    Returns: Velocity dispersion in km/s.
    """
    return 200 | units.kms * (mBH / (1.66e8 | units.MSun))**(1/4.86)

def get_vesc(mass):
    """
    Calculate galactic escape velocity assuming truncated isothermal sphere potential. 
    Default units in km/s.
    Args:
        Mass (units.mass): Mass of the BH in solar masses.
    Returns: Escape velocity in km/s.
    """
    vdisp = get_vdisp(mass)
    return 3. * vdisp

def get_Mgal_from_Mbh(mass, inverse=True):
    """
    Haring-Rix relation (BH-Galactic Bulge mass).
    Args:
        mass (units.mass): Mass of the galaxy in solar masses.
        inverse (bool):    If True, get galaxy mass from BH mass; else get BH mass from galaxy mass.
    Returns: Black hole mass in units.mass.
    """
    alpha_val = 8.8 + 1.24 * np.log10(mass/(1e11 | units.MSun))
    if inverse:  # Get BH mass from galaxy mass
        return 10**((np.log10(mass.value_in(units.MSun))-8.8)/1.24) * 10**11 | units.MSun
    return 10**alpha_val | units.MSun

def get_sphere_of_influence(mBH):
    """
    Get the sphere of influence radius for a black hole of mass mBH.
    Args:
        mBH (units.mass): Mass of the black hole in solar masses.
    Returns: Sphere of influence radius in units.length.
    """
    sigma = get_vdisp(mBH)
    return (constants.G * mBH / sigma**2)