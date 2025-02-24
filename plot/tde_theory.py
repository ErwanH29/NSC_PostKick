from amuse.lab import units, constants
import numpy as np


def dispersion_velocity(SMBH_mass):
    return 200 * (SMBH_mass/(1.66 * 10**8 | units.MSun))**(1/4.86) | units.kms

def sphere_of_influence(SMBH_mass):
    vdisp = dispersion_velocity(SMBH_mass)
    rinfl = constants.G*SMBH_mass/vdisp**2
    return rinfl
    
def frac_bound(gamma, SMBH_mass, vkick):
    rinfl = sphere_of_influence(SMBH_mass)
    fb = 11.6*gamma**-1.75*(constants.G*SMBH_mass/(rinfl*vkick**2))**(3-gamma)
    print(f"For gamma={gamma}, SMBH_mass={SMBH_mass}, vkick={vkick}, fbound={fb}, Mbound={fb*SMBH_mass}")
    return fb

def TDE_rate(gamma, SMBH_mass, vkick):
    """Eqn 5 of Komossa & Merritt 2008"""
    fbound = frac_bound(gamma, SMBH_mass, vkick)
    value = 6.5 * 10**-6 * (SMBH_mass/(10**7 | units.MSun))**-1 * (vkick/(10**3 | units.kms))**3 * fbound/10**-3 | (1/units.yr)
    print(f"For gamma={gamma}, SMBH_mass={SMBH_mass}, vkick={vkick}, TDE rate={value.in_(units.kyr**-1)}")

def TDE_timescale(SMBH_mass, vkick):
    """Between Eqn 5 and Eqn 6 of Komossa & Merritt 2008"""
    tau = 3.6 * constants.G*SMBH_mass**2/(vkick**3*(AVG_STAR_MASS))
    return tau

def TDE_rate_NR(SMBH_mass):
    """Eqn 38(a) of Wang & Merritt 2004"""
    vdisp = dispersion_velocity(SMBH_mass)
    rate = 4.2*10**-4 * (vdisp/(100 | units.kms))**(-1.15) | units.yr**-1
    #rate = 7.1*10**-4 * (vdisp/(70 | units.kms))**(3.5) * (SMBH_mass/(1e6 | units.MSun))**-1 | units.yr**-1
    return rate #* (SMBH_mass/(10**6 | units.MSun))**-0.46

def precession_tau(SMBH_mass, vkick, sma):
    """Eqn 1 ofRauch & Tremaine 1996"""
    fbound = frac_bound(1.75, SMBH_mass, vkick)
    HCSC_mass = SMBH_mass * fbound
    orb_period = 2*np.pi*sma/(np.sqrt(constants.G*SMBH_mass/sma))
    tau_prec = SMBH_mass/HCSC_mass * orb_period
    print(f"For SMBH_mass={SMBH_mass}, vkick={vkick}, RR timescale={tau_prec.in_(units.kyr)}")

def orb_period_max(SMBH_mass, vkick):
    """Eqn 2 of Rauch & Tremaine 1996"""
    sma_max = 8*constants.G*SMBH_mass/vkick**2
    orb_period_max = 2*np.pi*sma_max/(np.sqrt(constants.G*SMBH_mass/sma_max))
    print(f"For SMBH_mass={SMBH_mass}, vkick={vkick}, sma_max={sma_max.in_(units.pc)}, orb_period_max={orb_period_max.in_(units.kyr)}")

AVG_STAR_MASS = 3.8 | units.MSun

for mass in [10**5, 4*10**5] | units.MSun:
    for vkick in [300, 600] | units.kms:
        orb_period_max(mass, vkick)
        precession_tau(mass, vkick, sma=0.1 | units.pc)
        TDE_rate(SMBH_mass=mass, vkick=vkick, gamma=1.75)
        frac_bound(1.75, mass, vkick)
        print("============================")

print(TDE_timescale(4*10**5 | units.MSun, 300 | units.kms).in_(units.yr))
print(TDE_timescale(4*10**5 | units.MSun, 600 | units.kms).in_(units.yr))
print(TDE_rate_NR(4*10**5 | units.MSun).in_(units.kyr**-1))