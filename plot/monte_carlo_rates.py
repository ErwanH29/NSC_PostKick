import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
from scipy.interpolate import interp1d

from amuse.lab import units, constants


def event_rate(zeta, z_range, M_range, gg_mergers, N_event, phi, vkick, gamma):
    """
    Compute the event rate for a given set of parameters.
    """
    
    M_range_unitless = M_range.value_in(units.MSun)
    
    def integrand(M, z):
        """
        Compute the integrand using unitless inputs for scipy integration.
        """
        M_phys = M | units.MSun  # Galaxy masses
        gg_rate = gg_mergers(z, M_phys)  # Galaxy-galaxy merger rate
        phi_val = phi(z, M_phys).value_in(units.Mpc**-3)  # Press-Schechter. Convert to Mpc^-3
        event_count = N_event(M_phys, vkick, gamma)  # Event based on our fit
        
        return zeta**2 * gg_rate * phi_val * event_count  # Unitless
    
    integral, _ = integ.dblquad(integrand, z_range[0], z_range[-1], 
                                lambda z: M_range_unitless[0], lambda z: M_range_unitless[-1])
    
    return integral | units.Myr**-1  # Convert result back to AMUSE units

def N_event(M, vkick, gamma):
    """Compute event rate from fit. Units are per Myr"""
    AVG_STAR_MASS = 2.43578679652 | units.MSun
    
    SMBH_mass = haring_rix_relation(M)
    Rtide = (0.844**2 * SMBH_mass/AVG_STAR_MASS)**(1./3.) | units.RSun
    Rcluster = (8. * constants.G * SMBH_mass / vkick**2.)
    vdisp = 200 * (SMBH_mass/(1.66*10**8 | units.MSun))**(1/4.86) | units.kms
    rinfl = constants.G*SMBH_mass/vdisp**2
    
    C_RR =  1.6 * (SMBH_mass / AVG_STAR_MASS)**((gamma-1)/3) * (vkick/vdisp)**(-2*(gamma-1))
    ln_term = np.log(SMBH_mass / AVG_STAR_MASS) / np.log(Rcluster / Rtide)
    f_bound = 11.6 * gamma **-1.75 * (constants.G*SMBH_mass/(rinfl*vkick**2))**(3-gamma)
    kick_rcluster = (vkick/Rcluster).value_in(units.Myr**-1)
    
    Nevent = C_RR * ln_term * kick_rcluster * f_bound
    return Nevent

def recoil_kick():
    return 250 | units.kms  # PLACEHOLDER

def press_schechter(z, M):
    """Compute the Press-Schechter mass function at redshift z and mass M."""
    
    phi_star = phi_star_interp(z) | (units.Mpc)**-3
    mass_param = M / (M_star_interp(z) | units.MSun)
    alpha_param = alpha_interp(z)
    
    press_schechter = phi_star * (mass_param)**(alpha_param) * np.exp(-mass_param)
    return press_schechter

def N_gg_mergers(z, M):
    return 1e-3 * (1 + z) * np.exp(-M / (1e12 | units.MSun))  # PLACEHOLDER

def haring_rix_relation(mass):
    """Compute the Haring-Rix relation for a given mass."""
    alpha = 8.2 + 1.12 * np.log10(mass/(1e11 | units.MSun))
    return 10**alpha | units.MSun

zeta = 0.86
gamma = 1.75
galaxy_masses = np.linspace(1e8, 5e8, 100) | units.MSun
redshift_range = np.linspace(0, 4, 100)

redshift_bins = np.array([0.05, 0.3, 0.75, 1.5, 2.5, 3.5])
phi_star = 10**-3 * np.array([0.84, 0.84, 0.74, 0.45, 0.22, 0.12])  # In units 10**-3 Mpc^-3, Press-Schechter normalisation
M_norm = 10**np.array([11.14, 11.11, 11.06, 10.91, 10.78, 10.60])  # In MSun, Press-Schechter normalisation
alpha_values = np.array([-1.43, -1.45, -1.48, -1.57, -1.66, -1.74])  # Press-Schechter power-law
vkick = 300 | units.kms  # PLACEHOLDER

phi_star_interp = interp1d(redshift_bins, phi_star, kind='linear', fill_value='extrapolate')
M_star_interp = interp1d(redshift_bins, M_norm, kind='linear', fill_value='extrapolate')
alpha_interp = interp1d(redshift_bins, alpha_values, kind='linear', fill_value='extrapolate')

print("================= Event Rate =================")
print(f"Redshift range: [{redshift_bins[0]}, {redshift_bins[-1]}]")
print(f"Galaxy Mass range: [{galaxy_masses[0]}, {galaxy_masses[-1]}]")
print(f"BH Mass range: [{haring_rix_relation(galaxy_masses[0])}, {haring_rix_relation(galaxy_masses[-1])}]")
print(f"zeta: {zeta}")
print(f"Vkick: {vkick}")
print(f"Gamma: {gamma}")

gamma_event_result = event_rate(zeta, redshift_range, galaxy_masses, N_gg_mergers, N_event, press_schechter, vkick, gamma)
print(gamma_event_result)
