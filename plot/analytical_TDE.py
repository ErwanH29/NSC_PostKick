from amuse.lab import units, constants
import numpy as np
import matplotlib.ticker as mtick

def sphere_of_influence(SMBH_mass):
    """Extract sphere of influence"""
    vdisp = 200. * (SMBH_mass/(1.66 * 10**8 | units.MSun))**(1/4.86) | units.kms
    rinfl = constants.G*SMBH_mass/vdisp**2
    return rinfl

def cluster_mass(SMBH_mass, vkick):
    """Extract HCSC mass"""
    rinfl = sphere_of_influence(SMBH_mass)
    #rinfl = 35 * (SMBH_mass/(10**8 | units.MSun))**(0.56) | units.pc
    mcluster = 11.6*GAMMA**-1.75 * SMBH_mass \
                * (constants.G*SMBH_mass/(rinfl*vkick**2.))**(3. - GAMMA)
    return mcluster

def tidal_radius(SMBH_mass):
    """Extract tidal radius"""
    rstar = (AVG_STELLAR_MASS.value_in(units.MSun)) ** 0.8
    rtide = rstar*(0.844*SMBH_mass/(AVG_STELLAR_MASS))**(1./3.) | units.RSun
    return rtide

def stone_2017(SMBH_mass, vdisp, vkick, rcluster):
    stellar_tide = tidal_radius(SMBH_mass)
    mcluster = cluster_mass(SMBH_mass, vkick)
    
    number_density = mcluster/(4./3.*np.pi*rcluster**3.) * 1/(AVG_STELLAR_MASS)
    cross_section = np.pi*stellar_tide**2.*(1.+2.*constants.G*SMBH_mass/(stellar_tide*vdisp**2.))
    TDE_rate = number_density*cross_section*vdisp
    print(f"Stone 2017 TDE rate: MSMBH = {SMBH_mass.in_(units.MSun)}, vkick = {vkick.in_(units.kms)}, {TDE_rate.in_(1/units.kyr)}")

def rizzuto_2023(SMBH_mass, vdisp, vkick, rcluster):
    mcluster = cluster_mass(SMBH_mass, vkick)
    fbound = 1  # Fraction of stars within the sphere of influence bound to SMBH (here rHCSC)
    density = mcluster/(4./3.*np.pi*rcluster**3.)
    
    TDE_rate = 1.1*0.8*fbound * np.log(0.22*SMBH_mass/AVG_STELLAR_MASS) \
                * (SMBH_mass/(10**3 | units.MSun)) * (density/(10**7 | units.MSun/units.pc**3)) \
                    * (vdisp/(100 | units.kms))**-3 | units.Myr**-1
    print(f"Rizzuto et al. 2023 TDE rate: MSMBH = {SMBH_mass.in_(units.MSun)}, vkick = {vkick.in_(units.kms)}, {TDE_rate.in_(1/units.kyr)}")

def komossa_2008(SMBH_mass, vdisp, vkick, rcluster):
    stellar_tide = tidal_radius(SMBH_mass)
    RCLUSTER = constants.G*SMBH_mass/vkick**2
    fbound = cluster_mass(SMBH_mass, vkick) / SMBH_mass
    mcluster = cluster_mass(SMBH_mass, vkick)
    C_RR = 0.14  # NEED BETTER VALUE --> order of magnitude off when we base off Stone
    lnL = np.log(SMBH_mass/AVG_STELLAR_MASS)
    lnR = np.log(RCLUSTER/stellar_tide)
    
    TDE_rate = C_RR * lnL/lnR * (vkick/RCLUSTER) * fbound
    print(f"Komossa & Merritt 2008 TDE rate: MSMBH = {SMBH_mass.in_(units.MSun)}, vkick = {vkick.in_(units.kms)}, {TDE_rate.in_(1/units.kyr)}")
    TDE_rate = 6.5*10**-3 * (SMBH_mass/(10**7 | units.MSun))**-1 * (vkick/(1000|units.kms))**3 * (fbound/10**-3) | units.kyr**-1
    print(f"Komossa & Merritt 2008 TDE rate: MSMBH = {SMBH_mass.in_(units.MSun)}, vkick = {vkick.in_(units.kms)}, {TDE_rate.in_(1/units.kyr)}")
    
    N = cluster_mass(SMBH_mass, vkick)/AVG_STELLAR_MASS
    TDE_rate = np.pi/np.sqrt(2) * N*constants.G**2 \
                * AVG_STELLAR_MASS * lnL/(vdisp**3 * lnR) \
                   * (SMBH_mass+mcluster)/(4/3 * np.pi * rcluster**3)
    TDE_rate *= SMBH_mass/(N*AVG_STELLAR_MASS)
    print(f"Modified Komossa & Merritt 2008 TDE rate: MSMBH = {SMBH_mass.in_(units.MSun)}, vkick = {vkick.in_(units.kms)}, {TDE_rate.in_(1/units.kyr)}")
    
    N = cluster_mass(SMBH_mass, vkick)/AVG_STELLAR_MASS
    fbound = cluster_mass(SMBH_mass, vkick) / SMBH_mass
    volume = 4/3 * np.pi * rcluster**3
    TDE_rate = N/(lnR * fbound) * (np.pi * constants.G**2 * AVG_STELLAR_MASS * mcluster * lnL)/(np.sqrt(2) *vdisp**3 * volume)
    TDE_rate *= SMBH_mass/(N*AVG_STELLAR_MASS)
    print(f"Modified Komossa & Merritt 2008 TDE rate: MSMBH = {SMBH_mass.in_(units.MSun)}, vkick = {vkick.in_(units.kms)}, {TDE_rate.in_(1/units.kyr)}")
    
    density = 2.6e7 | units.MSun/units.pc**3  # Schodel (2018) from MW
    unbound_rate = 2*np.pi*constants.G*SMBH_mass*(density/AVG_STELLAR_MASS) * vkick**-1 * stellar_tide
    print(f"Unbound rate: MSMBH = {SMBH_mass.in_(units.MSun)}, vkick = {vkick.in_(units.kms)}, {unbound_rate.in_(1/units.kyr)}")
    
def wang_2004(SMBH_mass, gamma):
    """Extract TDE rate for NSC (unrecoiled)"""
    alpha = (27-19*gamma)/(6*(4-gamma))
    vdisp = 200 * (SMBH_mass/(1.66 * 10**8 | units.MSun))**(1/4.86) | units.kms
    #vdisp = 100 | units.kms
    #vdisp = 200 * (SMBH_mass/(1.48 * 10**8 | units.MSun))**(1/4.65) | units.kms
    TDE_rate = 7.1*10**-4*(vdisp/(70 | units.kms))**3.5*(SMBH_mass/(10**6 | units.MSun))**alpha | units.yr**-1
    return TDE_rate

def our_formula(SMBH_mass, vkick, gamma, rinfl=None):
    #vkick = 4000 | units.kms
    #SMBH_mass = 1e6 | units.MSun
    
    vdisp = 200 * (SMBH_mass/(1.66 * 1e8 | units.MSun))**(1/4.86) | units.kms
    if rinfl is None:
        rinfl = constants.G*SMBH_mass/(vdisp**2)
        
    rkick = 8. * constants.G*SMBH_mass/vkick**2
    AVG_STAR_MASS = 2.43578679652 | units.MSun
    rtide = (0.844**2 * SMBH_mass/AVG_STAR_MASS)**(1./3.) | units.RSun
    
    term1 = 0.14 * (SMBH_mass/AVG_STAR_MASS)**((gamma-1)/3) * (vkick/vdisp)**(-2*(gamma-1))
    term2 = np.log(SMBH_mass/AVG_STAR_MASS) / np.log(rkick/rtide)
    term3 = (vkick/rkick).value_in(units.Myr**-1)
    term4 = 11.6*gamma**-1.75 * (constants.G*SMBH_mass/(rinfl*vkick**2.))**(3.-gamma)
    
    term_t = term1 * term2 * term3 * term4
    formula = 31.188711107801634 * term_t # in 1/Myr
    #print(f"Our formula TDE rate: MSMBH = {SMBH_mass.in_(units.MSun)}, g = {gamma}, vkick = {vkick.in_(units.kms)}, {formula.value_in(units.kyr**-1)}")
    return formula

import matplotlib.pyplot as plt
import matplotlib

AVG_STELLAR_MASS = 3.07 | units.MSun
GAMMA = 1.75
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

cmap = matplotlib.colormaps['cool']
colours = cmap(np.linspace(0, 1, 5))
labels = [
    r"$10^5\ {\rm M_{\odot}}$", 
    r"$4 \times\ 10^5 {\rm M_{\odot}}$", 
    r"$10^6\ {\rm M_{\odot}}$", 
    r"$4 \times 10^6\ {\rm M_{\odot}}$", 
    r"$10^7\ {\rm M_{\odot}}$"
]

gamma = np.linspace(1, 2, 2000)

fig, ax = plt.subplots(figsize=(6,5))
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.fill_between(gamma, 1e6*(1e-5), 1e6*(1e-4), color="pink", alpha=0.5)
ax.fill_between(gamma, 1e6*(1e-4), 1e6*(10**-3.5), color="dodgerblue", alpha=0.15)

gamma = np.linspace(0.5, 2., 10000)
min_diff = abs(gamma - 1.75).argmin()
for i,mass in enumerate([1e5, 4e5, 1e6, 4e6, 1e7]):
    mass = mass | units.MSun
    vdisp = 200 * (mass/(1.66 * 10**8 | units.MSun))**(1/4.86) | units.kms
    rate_300 = [ ]
    for g in gamma:
        rate = our_formula(mass, 300 | units.kms, gamma=g)
        rate_300.append(rate)
        if abs(g - 1.75) < 0.00008:
            print(mass.value_in(units.MSun)/10**5, g, rate)
        elif abs(g - 1.0) < 0.00008:
            print(mass.value_in(units.MSun)/10**5, g, rate)
        
    rate_300 = np.array(rate_300)
    
    int_1e2 = abs(rate_300 - 1e2).argmin()
    int_1e1 = abs(rate_300 - 1e1).argmin()
    
    print(f"Mass: {mass}")
    print(f"Us at Gamma = 1.75, {rate_300[min_diff]*1e-6}")
    print(f"Intersection 1e2: {gamma[int_1e2]}")
    print(f"Intersection 1e1: {gamma[int_1e1]}")
    
    gamma_1e2 = gamma[int_1e2]
    gamma_1e1 = gamma[int_1e1]
    ax.plot(gamma, rate_300, color=colours[i], zorder=1, lw=2)
    ax.scatter(None, None, color=colours[i], label=labels[i])

rate = [ ]
for g in gamma:
    rate.append(our_formula(1e7 | units.MSun, 1000 | units.kms, gamma=g, rinfl=10|units.pc))
    
ax.plot(gamma, rate, color=colours[i], zorder=1, lw=2, ls="--")
    
ax.set_xlim(1., 2.)
ax.set_ylim(1.01, 1e4)
ax.set_xlabel(r"$\gamma$", fontsize=14)#
ax.set_ylabel(r"$\dot{N}$ [Myr$^{-1}$]", fontsize=14)
ax.legend(fontsize=14, loc="lower right")
ax.set_yscale("log")
ax.legend(fontsize=14)
ax.tick_params(axis="y", which='both', 
                direction="in", 
                labelsize=14)
ax.tick_params(axis="x", which='both', 
                direction="in", 
                labelsize=14)
plt.savefig(f"plot/figures/wang_plot.pdf", dpi=300, bbox_inches='tight')
plt.clf()
STOP

wang_2004(SMBH_mass=1e7 | units.MSun)
wang_2004(SMBH_mass=1e5 | units.MSun)
wang_2004(SMBH_mass=4e5 | units.MSun)
our_formula(SMBH_mass=1e5 | units.MSun, vkick=300 | units.kms)

# MSMBH =  1e5,    4e5,   1e5,   4e5
# vKICK =  300,    300,   600,   600
# vdisp =  180,    190,   300,   330    BASED ON PLOTS OF LONG-TERM
# rclus = 0.02,  0.045,  0.01,  0.02  BASED ON PLOTS OF LONG-TERM

#                    MSMBH,          vkick,            vdisp,          rcluster
configs = {
    "Model_A": [1e5 | units.MSun, 300 | units.kms, 180 | units.kms, 0.0157 | units.pc],
    "Model_B": [4e5 | units.MSun, 300 | units.kms, 190 | units.kms, 0.0449 | units.pc],
    "Model_C": [1e5 | units.MSun, 600 | units.kms, 300 | units.kms, 0.0074 | units.pc],
    "Model_D": [4e5 | units.MSun, 600 | units.kms, 330 | units.kms, 0.0193 | units.pc]
}    

model_choice = "Model_B"    
rcluster = constants.G * configs[model_choice][0] / configs[model_choice][2]**2
ratio_r = rcluster/configs[model_choice][-1]  # Merritt / Us
ratio_v = configs[model_choice][1] / configs[model_choice][2]   # Merritt / Us
prop = ratio_r**2.5*ratio_v**-2
print("Merritt-to-us-ratio: ", prop)

stone_2017(SMBH_mass=configs[model_choice][0],
           vdisp=configs[model_choice][2],
           vkick=configs[model_choice][1],
           rcluster=configs[model_choice][3])
rizzuto_2023(SMBH_mass=configs[model_choice][0],
           vdisp=configs[model_choice][2],
           vkick=configs[model_choice][1],
           rcluster=configs[model_choice][3])
komossa_2008(SMBH_mass=configs[model_choice][0],
             vdisp=configs[model_choice][2],
             vkick=configs[model_choice][1],
             rcluster=configs[model_choice][3])
wang_2004(SMBH_mass=configs[model_choice][0])