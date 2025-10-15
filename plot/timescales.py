from amuse.lab import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import numpy as np

avg_star_mass = 1 | units.MSun
def check_full_LC(gamma, MBH, vkick):
    rtidal = (0.844**2 * MBH/avg_star_mass)**(1/3) | units.AU
    Lmin = np.sqrt(2*constants.G*MBH*rtidal)
    Lmax = np.sqrt(2*constants.G*MBH*get_rkick(MBH, vkick))
    term_1 = Lmin / Lmax
    
    eta_s = 1.4
    N = get_MHCSC(MBH, gamma, vkick) / avg_star_mass
    term_2 = eta_s * avg_star_mass * N**0.5 / MBH
    if term_1 < term_2:
        print("Full Loss Cone")
    else:
        print("Empty Loss Cone")

def get_density(MBH, vkick, gamma):
    rkick = get_rkick(MBH, vkick)
    mHCSC = get_MHCSC(MBH, gamma, vkick)
    rho0_r0 = (3-gamma)/(4*np.pi) * mHCSC * rkick**(gamma-3)
    density = rho0_r0 * rkick**-gamma
    return density

def get_kepler_period(MBH, vkick):
    rkick = get_rkick(MBH, vkick)
    period = np.sqrt(4*np.pi**2*rkick**3/(constants.G*MBH))
    return period

def get_rkick(MBH, vkick):
    return (8*constants.G*MBH)/(vkick**2)

def get_rinfl(MBH):
    sigma = 200 * (MBH/(1.66e8 | units.MSun))**(1/4.86) | (units.kms)
    rinfl = constants.G * MBH / sigma**2
    return rinfl

def get_MHCSC(MBH, gamma, vkick):
    rinfl = get_rinfl(MBH)
    HCSC_mass = 11.6*gamma**-1.75 * MBH * (constants.G * MBH/(rinfl * vkick**2))**(3-gamma)
    return HCSC_mass

def get_tNR(MBH, gamma, vkick):
    F1 = 11.6*gamma**-1.75
    rkick = get_rkick(MBH, vkick)
    rinfl = get_rinfl(MBH)
    
    term_1 = (MBH/avg_star_mass)**2
    term_2 = np.sqrt(4*np.pi**2*rkick**3/(constants.G*MBH))
    term_3 = (avg_star_mass * (F1 * MBH * (constants.G*MBH/(rinfl*vkick**2))**(3-gamma))**-1)
    t = term_1 * term_2 * term_3
    print("NR Time: ", t.in_(units.Myr))
    
    mHCSC = get_MHCSC(MBH, gamma, vkick)
    numerator = MBH**2
    denominator = avg_star_mass * mHCSC * np.log(MBH/avg_star_mass)
    t = numerator/denominator * get_kepler_period(MBH, vkick)
    print("NR Time (Alt 2): ", t.in_(units.Myr))
    return t
    
def get_tRR(MBH, gamma, vkick):
    F1 = 11.6*gamma**-1.75
    rkick = get_rkick(MBH, vkick)
    sigma = 200 * (MBH/(1.66e8 | units.MSun))**(1/4.86) | (units.kms)
    rinfl = constants.G * MBH/sigma**2
    
    term_1 = (MBH/avg_star_mass)**2
    term_2 = np.sqrt(4*np.pi**2*rkick**3/(constants.G*MBH))
    term_3 = (avg_star_mass * (F1 * MBH * (constants.G*MBH/(rinfl*vkick**2))**(3-gamma))**-1)
    term_4 = MBH * term_3 / avg_star_mass * term_2
    t = term_1 * term_2**2 * term_3 * term_4**-1
    print("RR Time: ", t.in_(units.Myr))
    
    t = MBH / avg_star_mass * get_kepler_period(MBH, vkick)
    print("RR Time (Alt 2): ", t.in_(units.Myr))
    return t
    
def get_tGR(MBH, vkick, gamma):
    rkick = get_rkick(MBH, vkick)
    Nsyst = get_MHCSC(MBH, gamma, vkick) / avg_star_mass
    ang_circ = np.sqrt(constants.G * MBH * rkick)
    ang_ISCO = 4 * constants.G * MBH / constants.c
    porb = get_kepler_period(MBH, vkick)
    t_GR = 3/8 * (MBH/avg_star_mass)**2 * (ang_ISCO/ang_circ)**2 * porb/Nsyst
    print(f"t_GR={t_GR.in_(units.Myr)}")
    return t_GR

def get_tcross(MBH, vkick):
    rkick = get_rkick(MBH, vkick)
    t = rkick/vkick
    print(f"tcross={t.in_(units.yr)}")
    return t

def get_tprec(MBH, vkick, gamma):
    t_orb = get_kepler_period(MBH, vkick)
    HCSC_mass = get_MHCSC(MBH, gamma, vkick)
    t_prec = MBH/HCSC_mass * t_orb
    print(f"t_prec={t_prec.in_(units.Myr)}")
    return t_prec


MBH_arr = [1e5, 4e5] | units.MSun
gamma = 1.75
vkick_arr = np.linspace(100, 1000, 1000) | units.kms

cmap = matplotlib.colormaps['cool']
black_rgba = colors.to_rgba("black")
carray = cmap(np.linspace(0.1, 1, 5))
labels = [
    r"$t_{NR}$",
    r"$t_{RR}$",
    r"$t_{GR}$",
    r"$t_{\omega}$",
]

ls = ["--", "-", "-."]
for MBH in MBH_arr:
    for ig, gamma in enumerate([1.0, 1.75]):
        data = {
            "t_NR": [],
            "t_RR": [],
            "t_GR": [],
            "t_prec": [],
        }
        for vkick in vkick_arr:
            print(f"For MBH={MBH.in_(units.MSun)}, vkick={vkick.in_(units.kms)}")
            check_full_LC(gamma, MBH, vkick)
            rinfl = get_rinfl(MBH)
            print(f"rinfl={(rinfl/vkick).in_(units.yr)}")
            N_HCSC = get_MHCSC(MBH, gamma, vkick) / avg_star_mass
            data["t_NR"].append(N_HCSC/get_tNR(MBH, gamma, vkick).value_in(units.yr))
            data["t_RR"].append(N_HCSC/get_tRR(MBH, gamma, vkick).value_in(units.yr))
            data["t_GR"].append(N_HCSC/get_tGR(MBH, vkick, gamma).value_in(units.yr))
            data["t_prec"].append(N_HCSC/get_tprec(MBH, vkick, gamma).value_in(units.yr))
            print("="*50)

        for ik, key in enumerate(data.keys()):
            data[key] = np.array(data[key])
            plt.plot(vkick_arr.value_in(units.kms), data[key], label=key if ig == 1 else None, color=carray[ik], ls=ls[ig], lw=1+2*ig)
    plt.yscale("log")
    plt.ylabel(r"$\Gamma$ [yr^{-1}]")
    plt.xlabel(r"$v_{kick}$ [km/s]")
    plt.legend(labels)
    plt.show()