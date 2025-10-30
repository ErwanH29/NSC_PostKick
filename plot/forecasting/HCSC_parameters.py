import numpy as np
from amuse.units import units, constants
from plot.forecasting.forecast_parameters import AVG_STAR_MASS, AVG_STAR_RAD, SWITCH_FACTOR
from plot.forecasting.gal_BH_relations import get_sphere_of_influence

def get_rkick(mBH, vkick):
    """
    Get the kick radius for a recoiling black hole.
    Args:
        mBH (units.mass):        Mass of the black hole in solar masses.
        vkick (units.velocity):  Kick velocity in km/s.
    Returns: Kick radius.
    """
    return (8 * constants.G * mBH / vkick**2)

def get_mHCSC(mBH, vkick, gamma, rSOI):
    """
    Get the mass of the hypercompact stellar cluster (HCSC) bound to the recoiling black hole.
    Args:
        mBH (units.mass):        Mass of the black hole in solar masses.
        vkick (units.velocity):  Kick velocity in km/s.
        gamma (float):           Power-law index for the mass function.
        rSOI (units.length):     Sphere of influence radius in parsecs.
    Returns:
        Mass of the HCSC in solar masses.
    """
    rSOI = get_sphere_of_influence(mBH)
    term1 = 11.6 * gamma**-1.75 * mBH
    term2 = (constants.G * mBH / (rSOI * vkick**2))**(3 - gamma)
    return term1 * term2

def get_Porb(mBH, r):
    """
    Get the Keplerian orbital period at radius r around a black hole of mass mBH.
    Args:
        mBH (units.mass): Mass of the black hole in solar masses.
        r (units.length): Radius in parsecs.
    Returns:
        Orbital period in years.
    """
    return 2 * np.pi * np.sqrt(r**3 / (constants.G * mBH))

def gamma_tau(age, M_bh, vkick, gamma, f_dep=0.5, RR=False):
    """
    Compute event rate from fit. 
    Total events assuming exhausted after 20 Myr.
    Args:
        age (units.time):        Age of HCSC.
        M_bh (units.mass):       Mass of the SMBH in solar masses.
        vkick (units.velocity):  Kick velocity in km/s.
        gamma (float):           Power-law index for the mass function.
        f_dep (float):           Fraction of stars that collide.
        RR (boolean):            Whether to calculate purely assuming RR.
    Returns:
        Total number of events.
    """
    rtide = (AVG_STAR_RAD.value_in(units.RSun)) * (0.844**2 * M_bh/AVG_STAR_MASS)**(1./3.) | units.RSun
    rSOI  = get_sphere_of_influence(M_bh)
    aGW   = 2e-4 * rSOI
    mHCSC = get_mHCSC(M_bh, vkick, gamma, rSOI)
    rkick = get_rkick(M_bh, vkick)
    Nclst = mHCSC / AVG_STAR_MASS

    if RR:
        CRR = 0.14
        term1 = np.log(M_bh/AVG_STAR_MASS)
        term2 = np.log(rkick/rtide)
        term3 = (vkick/rkick)
        term4 = mHCSC / M_bh  # Remaining mass of HCSC
        tau_RR = 3.6 * constants.G * M_bh**2 / (vkick**3 * AVG_STAR_MASS)  # From KM08
        Gamma0_RR = CRR * term1/term2 * term3 * term4

        Nlost = Gamma0_RR * tau_RR * (1.0 - np.exp(-age/tau_RR))
        if Nlost > f_dep * Nclst:  # Cluster exhausted by observational time
            return 0 | units.yr**-1
        
        return Gamma0_RR * np.exp(-(age - t_switch)/tau_RR)

    ### Burst Phase ###
    # Coeff absorbs factor of mClump, zeta (rinfl = zeta a_GW), beta for a_i vs. a_clump, k for RHill, ecc_phi for interaction time
    coeff = 2.61841299e+05
    alpha = 6.41880928e-04
    beta  = 9.55159492e+00

    term_a = (3-gamma) * constants.G * M_bh**2 * rtide**0.5
    term_b = (8*rSOI)**(gamma-3) * vkick**-1
    term_c = aGW**(-(gamma-0.5)) * ((2*((2*gamma+3)*((2*gamma+1) - 4*gamma + 2) + 4*gamma**2 - 1))/((2*gamma - 1)*(2*gamma + 1)*(2*gamma + 3)))
    term_d = alpha/(rSOI**(3/2)/(np.sqrt(constants.G * M_bh)) * (beta)**(gamma-3))
    term_e = 1/(M_bh.value_in(units.MSun)**(1/3) * np.sqrt(constants.G*M_bh)) * (aGW)**(3/2)
    Gamma0_burst = coeff * term_a * term_b * term_c * term_d * term_e / AVG_STAR_MASS * (1/1.3)

    ### Compute timescales
    tau_sec = (rSOI**(3/2)/(np.sqrt(constants.G * M_bh)) * (beta)**(gamma-3)) / alpha
    t_exhaust = Nclst/Gamma0_burst
    t_switch  = SWITCH_FACTOR * t_exhaust
    if age < t_switch:  # Purely burst
        Nlost = Gamma0_burst * tau_sec * (1.0 - np.exp(-age/tau_sec))
        if Nlost > f_dep * Nclst:  # Cluster exhausted more than allowed
            return 0 | units.yr**-1
        return Gamma0_burst * np.exp(-age/tau_sec)
    else:  # RR phase at measurement time -- Don't account for puffing up
        Nlost_burst = Gamma0_burst * tau_sec * (1.0 - np.exp(-t_switch/tau_sec))
        CRR = 0.14
        term1 = np.log(M_bh/AVG_STAR_MASS)
        term2 = np.log(rkick/rtide)
        term3 = (vkick/rkick)
        term4 = (mHCSC - Nlost_burst * AVG_STAR_MASS) / M_bh  # Remaining mass of HCSC
        tau_RR = 3.6 * constants.G * M_bh**2 / (vkick**3 * AVG_STAR_MASS)  # From KM08
        Gamma0_RR = CRR * term1/term2 * term3 * term4
        
        Nlost = Nlost_burst + Gamma0_RR * tau_RR * (1.0 - np.exp(-(age - t_switch)/tau_RR))
        if Nlost > f_dep * Nclst:  # Cluster exhausted by observational time
            return 0 | units.yr**-1
        
        return Gamma0_RR * np.exp(-(age - t_switch)/tau_RR)