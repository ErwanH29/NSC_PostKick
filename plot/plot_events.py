import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
from scipy.interpolate import interp1d

from amuse.lab import units, constants

def esc_velocity(Mass):
    """Calculate galactic escape velocity. Default units in km/s."""
    vdisp = 200 * (Mass/(1.66*10**8 | units.MSun))**(1/4.86) | units.kms
    return 3 * vdisp

def event_rate(z_range, M_range, IMBH_IMBH_merger, N_event, press_schechter):
    """
    Compute the event rate
    
    Args:
        z_range: Redshift range
        M_range: Galaxy mass range
        IMBH_IMBH_merger: IMBH-IMBH merger rate function
        N_event: Event rate function
        press_schechter: Press-Schechter mass function
    Returns:
        Event rate in [Gyr^-1]
    """
    print("...Normalising PS function...")
    total_phi_dict = PS_normalisation(z_range, M_range, press_schechter)  # In Mpc^-3
    M_range_unitless = M_range.value_in(units.MSun)  # In MSun
    
    def get_total_phi(z):
        """Get the total number of objects at redshift z. In units Mpc^-3."""
        z_nearest = abs(z_range - z).argmin()
        z = z_range[z_nearest]
        return total_phi_dict[z]  # In Mpc^-3
    
    def integrand(v, M, z):
        """Compute the integrand using unitless inputs for scipy integration."""
        M_phys = M | units.MSun  # Galactic mass
        v_esc = esc_velocity(haring_rix_relation(M_phys))  # Galactic escape velocity
        
        # Kick probability
        Pkick = vkick_pdf(v)
        v = v | units.kms
        if v < v_esc:
            return 0
        
        # MGal probability
        ps_val = press_schechter(z, M_phys).value_in(units.Mpc**-3)  # In Mpc^-3
        ps_norm = get_total_phi(z)
        Pmass = ps_val / ps_norm  # Fraction of objects with mass M_phys at redshift z
        
        # IMBH-IMBH merger rate
        Rm = IMBH_IMBH_merger(z)
        Rm_Myr = Rm.value_in(units.Myr**-1)
        
        event_count = N_event(M_phys, v, GAMMA)  # Event based on our fit after 100 Myr
        
        return Pmass * ZETA**2 * Rm_Myr * Pkick * event_count * 1e-6  # Unitless
    
    integral, _ = integ.tplquad(
                    integrand, 
                    z_range[0], z_range[-1], 
                    lambda z: M_range_unitless[0], lambda z: M_range_unitless[-1],
                    lambda z, M: v_min, lambda z, M: v_max,
                    epsabs=1e-3, epsrel=1e-3
                    )
    
    return integral | units.Gyr**-1  # Convert result back to AMUSE units

def N_event(M, vkick, GAMMA):
    """Compute event rate from fit. Total events assuming exhausted after 20 Myr"""
    AVG_STAR_MASS = 2.43578679652 | units.MSun
    
    SMBH_mass = haring_rix_relation(M)
    Rtide = (0.844**2 * SMBH_mass/AVG_STAR_MASS)**(1./3.) | units.RSun
    Rcluster = (8. * constants.G * SMBH_mass / vkick**2.)
    vdisp = 200 * (SMBH_mass/(1.66*10**8 | units.MSun))**(1/4.86) | units.kms
    rinfl = constants.G*SMBH_mass/vdisp**2
    
    C_RR =  0.14 * (SMBH_mass / AVG_STAR_MASS)**((GAMMA-1)/0.56) * (vkick/vdisp)**(-0.3*(GAMMA-1))
    ln_term = np.log(SMBH_mass / AVG_STAR_MASS) / np.log(Rcluster / Rtide)
    f_bound = 11.6 * GAMMA **-1.75 * (constants.G*SMBH_mass/(rinfl*vkick**2))**(3-GAMMA)
    kick_rcluster = (vkick/Rcluster).value_in(units.Myr**-1)
    
    Nevent = 6*10**-6 * C_RR * ln_term * kick_rcluster * f_bound * 20.**(0.9)
    return Nevent

def IMBH_IMBH_mergers(z):
    """IMBH-IMBH merger rate in [yr^-1] from arXiv:2412.15334"""
    merger_rate = merger_rate_interp(z) | units.yr**-1 * (units.Gpc)**-3 
    H0 = 67.4 | (units.kms/units.Mpc)
    OmegaM = 0.303
    OmegaLambda = 0.697
    
    # Using arXiv:9905116
    Ez = np.sqrt(OmegaM*(1+z)**3 + OmegaLambda)
    DM = (Ez * (1+z))**-1 * constants.c/H0

    # Integrate over comoving volume
    Da = DM/(1+z)**2
    dV = 4*np.pi * constants.c/H0 * (1+z)**2 * Da**2/Ez
    dN = merger_rate * dV
    
    return dN


def haring_rix_relation(mass):
    """Haring-Rix relation (BH-Galactic Bulge mass)"""
    alpha = 8.8 + 1.24 * np.log10(mass/(1e11 | units.MSun))
    return 10**alpha | units.MSun

def vkick_pdf(v):
    """
    Returns the probability density for vkick (with v in km/s).
    """
    return pdf_interp(v)

def PS_normalisation(z_range, M_range, press_schechter):
    """Normalise Press-Schechter mass function for a given redshift range and galaxy mass range. In units Mpc^-3."""
    M_range_unitless = M_range.value_in(units.MSun)
    
    total_phi_dict = { }
    for z in z_range:
        def integrand_mass(M):
            M_phys = M | units.MSun
            return press_schechter(z, M_phys).value_in(units.Mpc**-3)

        val, err = integ.quad(integrand_mass, M_range_unitless[0], M_range_unitless[-1])
        total_phi_dict[z] = val
    
    return total_phi_dict

def press_schechter(z, M):
    """Compute the Press-Schechter mass function at redshift z and mass M. In units Mpc^-3."""
    phi_star = phi_star_interp(z) | (units.Mpc)**-3
    mass_param = M / (M_star_interp(z) | units.MSun)
    alpha_param = alpha_interp(z)
    
    press_schechter = phi_star * (mass_param)**(alpha_param) * np.exp(-mass_param)
    return press_schechter

TDE_FACTOR = 2/3
GW_FACTOR = 1/3 * 0.13
NSC_FRAC = 1.0
ZETA = 1.0
GAMMA = 1.75

# Kick probability: arXiv:1201.1923
# vkick = 130 km/s no ejection for our SMBH range --> neglect this range
Prob_Distr ={
    "Kick Lower Limit": [100, 200, 300, 400, 500, 1000],#, 1500, 2000, 2500, 3000, 3500, 4000],
    "Kick< CDF":  [0.211364, 0.116901, 0.078, 0.05759, 0.140283]#4.0183]#, 1.0309, 0.2407, 0.0296, 0.0032, 0.0002]  # For hot mergers
}

vkick_bins = np.array(Prob_Distr["Kick Lower Limit"]) 
pdf_values = np.array(Prob_Distr["Kick< CDF"]) 
bin_midpoints = (vkick_bins[:-1] + vkick_bins[1:]) / 2
pdf_interp = interp1d(bin_midpoints, pdf_values, kind='linear', fill_value="extrapolate")
v_min, v_max = vkick_bins[0], vkick_bins[-1]


galaxy_masses = np.linspace(8.63e7, 3.15e8, 100) | units.MSun
redshift_range = np.linspace(0, 4, 100)

# Best fit parameters for Press-Schecter function. arXiv:1410.3485
merger_rate = [0.0006, 0.0006, 0.0006, 0.0035, 0.004, 0.0025]
redshift_bins = np.array([0.05, 0.3, 0.75, 1.5, 2.5, 3.5])
phi_star = 10**-3 * np.array([0.84, 0.84, 0.74, 0.45, 0.22, 0.12])  # In units 10**-3 Mpc^-3, Press-Schechter normalisation
M_norm = 10**np.array([11.14, 11.11, 11.06, 10.91, 10.78, 10.60])  # In MSun, Press-Schechter normalisation
alpha_values = np.array([-1.43, -1.45, -1.48, -1.57, -1.66, -1.74])  # Press-Schechter power-law

phi_star_interp = interp1d(redshift_bins, phi_star, kind='linear', fill_value='extrapolate')
M_star_interp = interp1d(redshift_bins, M_norm, kind='linear', fill_value='extrapolate')
alpha_interp = interp1d(redshift_bins, alpha_values, kind='linear', fill_value='extrapolate')
merger_rate_interp = interp1d(redshift_bins, merger_rate, kind='linear', fill_value='extrapolate')

print("================= Event Rate =================")
print(f"Redshift range: [{redshift_range[0]}, {redshift_range[-1]}]")
print(f"Galaxy Mass range: [{galaxy_masses[0]}, {galaxy_masses[-1]}]")
print(f"BH Mass range: [{haring_rix_relation(galaxy_masses[0])}, {haring_rix_relation(galaxy_masses[-1])}]")
print(f"ZETA: {ZETA}")
print(f"GAMMA: {GAMMA}")

GAMMA_event_result = event_rate(
                        z_range=redshift_range, 
                        M_range=galaxy_masses, 
                        IMBH_IMBH_merger=IMBH_IMBH_mergers, 
                        N_event=N_event, 
                        press_schechter=press_schechter
                        )
print((TDE_FACTOR * GAMMA_event_result).in_(units.yr**-1))
print((GW_FACTOR * GAMMA_event_result).in_(units.yr**-1))