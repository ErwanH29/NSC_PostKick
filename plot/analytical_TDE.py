from amuse.lab import *
import numpy as np
import matplotlib.pyplot as plt

from plot.plot_class import SetupFig

MSTAR = 1 | units.MSun
RSTAR = 1 | units.RSun
eta = 0.844


def get_mHCSC(mBH, vkick, gamma, rSOI=None):
    """
    Gets the mass of the hyper-compact stellar cluster (HCSC)
    Args:
        mBH (mass):        Mass of the black hole
        vkick (velocity):  Kick velocity of the black hole
        gamma (float):     Density profile of initial NSC
        rSOI (length):     Sphere of influence radius (optional)
    Returns: Mass of the HCSC
    """
    if rSOI is None:
        rSOI = get_sphere_of_influence(mBH)
    F1 = 11.6 * gamma **-1.75
    term = F1 * mBH * ((constants.G * mBH)/(vkick**2. * rSOI))**(3. - gamma)
    return term

def get_vdisp(mBH):
    """
    Gets the velocity dispersion within the sphere of influence
    Args:
        mBH (mass):  Mass of the black hole
    Returns: Velocity dispersion
    """
    return 200 | units.kms * (mBH / (1.66e8 | units.MSun))**(1/4.86)

def get_sphere_of_influence(mBH):
    """
    Gets the sphere of influence of the black hole
    Args:
        mBH (mass):  Mass of the black hole
    Returns: Sphere of influence radius
    """
    sigma = get_vdisp(mBH)
    return constants.G * mBH / sigma**2

def get_rkick(mBH, vkick):
    """
    Gets the initial HCSC radius
    Args:
        mBH (mass):  Mass of the black hole
        vkick (velocity):  Kick velocity of the black hole
    Returns: Radius of the HCSC
    """
    return (8 * constants.G * mBH) / (vkick**2)

def KM08_event_rate(mBH, vkick, gamma):
    """
    Gets the tidal disruption event rate from Komossa & Merritt (2008)
    Args:
        mBH (mass):  Mass of the black hole
        vkick (velocity):  Kick velocity of the black hole
        gamma (float):     Density profile of initial NSC
    Returns: TDE rate
    """
    fbound = get_mHCSC(mBH, vkick, gamma, rSOI=3 | units.pc) / mBH
    value = 6.5e-6 * (mBH/(1e7 | units.MSun))**(-1) * (vkick/(1000 | units.kms))**3 * fbound/10**-3 | units.yr**-1
    return value

def event_rate(vkick, mBH, gamma):
    """
    Gets the tidal disruption event rate from numerical results
    Args:
        vkick (velocity):  Kick velocity of the black hole
        mBH (mass):        Mass of the black hole
        gamma (float):     Density profile of initial NSC
    Returns: TDE rate
    """
    ### Coeff absorbs factor of:
    # mClump, zeta (rinfl = zeta a_GW), 
    # beta for a_i vs. a_clump, 
    # k for RHill, ecc_phi for interaction time

    coeff = 2.61841299e+05
    alpha = 6.41880928e-04
    beta  = 9.55159492e+00
    gamma = np.float128(gamma)

    rtide = RSTAR * (eta**2 * mBH / MSTAR)**(1/3)
    rSOI = get_sphere_of_influence(mBH)
    aGW = 2*10**-4 * rSOI

    term_a = (3-gamma) * constants.G * mBH**2 * rtide**0.5
    term_b = (8*rSOI)**(gamma-3) * vkick**-1
    term_c = aGW**(-(gamma-0.5)) * ((2*((2*gamma+3)*((2*gamma+1) - 4*gamma + 2) + 4*gamma**2 - 1))/((2*gamma - 1)*(2*gamma + 1)*(2*gamma + 3)))
    term_d = alpha/(rSOI**(3/2)/(np.sqrt(constants.G * mBH)) * (beta)**(gamma-3))
    term_e = 1/(mBH.value_in(units.MSun)**(1/3) * np.sqrt(constants.G*mBH)) * (aGW)**(3/2)
    value = coeff * term_a * term_b * term_c * term_d * term_e / MSTAR * (1/1.3)
    return value

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

gamma = np.linspace(0.51, 2., 1000)
vkick = [100, 1000]
MBH = [1e5, 1e6, 1e7] | units.MSun

plter = SetupFig()
fig, ax = plter.get_fig_ax(figsize=(6,5))
for im, mass in enumerate(MBH):
    for iv, v in enumerate(vkick):
        data_array = [[ ], [ ]]
        for ig, g in enumerate(gamma):
            rate_us = event_rate(v | units.kms, mass, g)
            rate_KM08 = KM08_event_rate(mass, v | units.kms, g)
            data_array[0].append(rate_us.value_in(units.yr**-1))
            data_array[1].append(rate_KM08.value_in(units.yr**-1))
        ax.plot(
            gamma, data_array[0], 
            color=plter.colours[im], 
            linestyle=plter.linestyles[iv], 
            lw=3-iv
            )
        if iv == 1 and im == 1:
            ax.plot(gamma, data_array[1], color=plter.colours[im], lw=1.5)

    exponent = int(np.floor(np.log10(mass.value_in(units.MSun))))
    mantissa = mass.value_in(units.MSun) / 10**exponent

    ax.scatter(
        [], [], color=plter.colours[im],
        label=rf"$10^{{{exponent}}}\,M_{{\odot}}$"
        )
ax.fill_between(gamma, (1e-5), (1e-4), color="pink", alpha=0.75)
ax.fill_between(gamma, (1e-4), (10**-3.5), color="dodgerblue", alpha=0.5)
ax.set_yscale("log")
ax.set_ylabel(r"$\Gamma$ [yr$^{-1}$]", fontsize=plter.TICK_SIZE)
ax.set_xlabel(r"$\gamma$", fontsize=plter.TICK_SIZE)
ax.set_xlim(gamma[0], gamma[-1])
ax.set_ylim(2e-7, 0.09)
ax.legend(fontsize=plter.TICK_SIZE)
plt.savefig("plot/figures/event_rate_vs_gamma.pdf", bbox_inches='tight')
plt.clf()