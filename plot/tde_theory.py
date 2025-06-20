from amuse.lab import units, constants
import numpy as np


def dispersion_velocity(SMBH_mass):
    """
    Calculate the dispersion velocity of stars in a cluster
    Args:
        SMBH_mass (units.mass): Mass of the SMBH
    Returns:
        vdisp (units.velocity): Dispersion velocity in km/s
    """
    norm_mass = 1e8 | units.MSun
    return 200.*(SMBH_mass/(1.66*norm_mass))**(1./4.86) | units.kms

def sphere_of_influence(SMBH_mass):
    """
    Calculate the sphere of influence of a SMBH
    Args:
        SMBH_mass (units.mass): Mass of the SMBH
    Returns:
        rinfl (units.length): Sphere of influence radius in parsecs
    """
    vdisp = dispersion_velocity(SMBH_mass)
    rinfl = constants.G*SMBH_mass/vdisp**2
    return rinfl
    
def frac_bound(gamma, SMBH_mass, vkick):
    """
    Calculate the fraction of stars bound to the SMBH
    Args:
        gamma (float): Power-law index of the density profile
        SMBH_mass (units.mass): Mass of the SMBH
        vkick (units.velocity): Kick velocity of the ejected stars
    Returns:
        fb (float): Stellar mass-to-SMBH mass ratio of bound stars
    """
    rinfl = sphere_of_influence(SMBH_mass)
    fb = 11.6*gamma**-1.75*(constants.G*SMBH_mass/(rinfl*vkick**2.))**(3.-gamma)
    print(f"For gamma={gamma}", end=", ")
    print(f"SMBH_mass={SMBH_mass}", end=", ")
    print(f"vkick={vkick}", end=", ")
    print(f"fbound={fb}", end=", ")
    print(f"Mbound={fb*SMBH_mass}")
    return fb

def TDE_rate(gamma, SMBH_mass, vkick):
    """
    Compute the TDE rate.
    Based on Eqn 5 of Komossa & Merritt 2008.
    Args:
        gamma (float): Power-law index of the density profile
        SMBH_mass (units.mass): Mass of the SMBH
        vkick (units.velocity): Kick velocity of the ejected stars
    """
    norm_mass = 1e7 | units.MSun
    norm_vel  = 1e3 | units.kms
    fbound = frac_bound(gamma, SMBH_mass, vkick)
    TDE_rate = 6.5e-6 * (SMBH_mass/norm_mass)**-1 * (vkick/norm_vel)**3 * fbound/10**-3 | (1/units.yr)
    print(f"For gamma={gamma}", end=", ")
    print(f"SMBH_mass={SMBH_mass}", end=", ")
    print(f"vkick={vkick}", end=", ")
    print(f"TDE rate={TDE_rate.in_(units.kyr**-1)}")

def TDE_timescale(SMBH_mass, vkick):
    """
    Compute the TDE timescale.
    Based on Eqn 5 and Eqn 6 of Komossa & Merritt 2008.
    Args:
        SMBH_mass (units.mass): Mass of the SMBH
        vkick (units.velocity): Kick velocity of the ejected stars
    Returns:
        tau (units.time): TDE timescale in years
    """
    tau = 3.6*constants.G*SMBH_mass**2/(vkick**3*(AVG_STAR_MASS))
    return tau

def TDE_rate_NR(SMBH_mass):
    """
    Compute the two-body relaxation induced TDE rate.
    Based on Eqn 38(a) of Wang & Merritt 2004.
    Args:
        SMBH_mass (units.mass): Mass of the SMBH
    Returns:
        rate (units.time**-1): TDE rate in inverse years
    """
    norm_vel = 100 | units.kms
    vdisp = dispersion_velocity(SMBH_mass)
    rate = 4.2e-4 * (vdisp/norm_vel)**(-1.15) | units.yr**-1
    return rate

def precession_tau(SMBH_mass, vkick, sma):
    """
    Compute the precession timescale for a star orbiting a SMBH.
    Based Eqn 1 of Rauch & Tremaine 1996.
    Args:
        SMBH_mass (units.mass): Mass of the SMBH
        vkick (units.velocity): Kick velocity of the ejected stars
        sma (units.length): Semi-major axis of the star's orbit
    """
    fbound = frac_bound(1.75, SMBH_mass, vkick)
    HCSC_mass = SMBH_mass * fbound
    orb_period = 2*np.pi*sma/(np.sqrt(constants.G*SMBH_mass/sma))
    tau_prec = SMBH_mass/HCSC_mass * orb_period
    print(f"For SMBH_mass={SMBH_mass}", end=", ")
    print(f"vkick={vkick}", end=", ")
    print(f"RR timescale={tau_prec.in_(units.kyr)}")

def orb_period_max(SMBH_mass, vkick):
    """
    Compute the orbital period at the outer-periphery of the HCSC.
    Eqn 2 of Rauch & Tremaine 1996.
    Args:
        SMBH_mass (units.mass): Mass of the SMBH
        vkick (units.velocity): Kick velocity of the ejected stars
    """
    sma_max = 8*constants.G*SMBH_mass/vkick**2
    orb_period_max = 2*np.pi*sma_max/(np.sqrt(constants.G*SMBH_mass/sma_max))
    print(f"For SMBH_mass={SMBH_mass}", end=", ")
    print(f"vkick={vkick}", end=", ")
    print(f"sma_max={sma_max.in_(units.pc)}", end=", ")
    print(f"orb_period_max={orb_period_max.in_(units.kyr)}")

AVG_STAR_MASS = 3.8 | units.MSun

for mass in [1e5, 4e5] | units.MSun:
    for vkick in [300, 600] | units.kms:
        orb_period_max(mass, vkick)
        precession_tau(mass, vkick, sma=0.1 | units.pc)
        TDE_rate(SMBH_mass=mass, vkick=vkick, gamma=1.75)
        frac_bound(1.75, mass, vkick)
        print("="*100)

print(TDE_timescale(4e5 | units.MSun, 300 | units.kms).in_(units.yr))
print(TDE_timescale(4e5 | units.MSun, 600 | units.kms).in_(units.yr))
print(TDE_rate_NR(4e5 | units.MSun).in_(units.kyr**-1))