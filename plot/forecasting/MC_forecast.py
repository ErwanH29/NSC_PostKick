import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid
import sys

from amuse.lab import units

from plot.forecasting.cosmological_functions import get_cosmic_time, R_gg, get_dV_dz
from plot.forecasting.forecast_parameters import TDE_FACTOR, merger_rate_IMBH, merger_rate_SMBH, Prob_Distr
from plot.forecasting.gal_BH_relations import get_vesc, get_Mgal_from_Mbh, get_Mbh_from_Mgal
from plot.forecasting.HCSC_parameters import gamma_tau
from plot.forecasting.sampling_functions import sample_mass_from_PS_at_z, sample_kick_velocity, sample_redshifts



COMP_KRITOS = False
NULL_RATE = 0. | (units.yr**-1)



def tickers(ax):
    """
    Function to setup axis
    Args:
        ax (axis):  Axis needing cleaning up
    Returns:
        ax (axis):  The cleaned axis
    """
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())

    ax.tick_params(axis="y", which='both', 
                    direction="in", 
                    labelsize=14)
    ax.tick_params(axis="x", which='both', 
                    direction="in", 
                    labelsize=14)
    return ax

def compute_rate_density(z_obs, gal_mass, N_zform_samp, N_HCSC_samp, zmax, gamma, Rm_gg, RR=False):
    """
    Compute the TDE/GW event rate up to redshift zmax. [yr^-1 Mpc^-3]
    Default zmax is set to 8.0 following Furlong et al. 2015.
    Args:
        z_obs (float):       Redshift up to which to compute the rate.
        gal_mass (list):     List of galaxy mass ranges in solar masses.
        gamma (float):       Power-law index for initian NSC density profile.
        zmax (float):        Maximum redshift for integration.
        N_zform_samp (int):  Number of samples for redshift formation sampling.
        N_HCSC_samp (int):   Number of samples for Monte Carlo integration.
        Rm_gg (function):    Interpolator for the BH-BH merger rate.
        RR (boolean):        Whether to calculate purely assuming RR.
    Returns: Computed event rate in yr^-1.
    """
    t_obs = get_cosmic_time(z_obs)

    if (COMP_KRITOS): ## Validation against Kritos et al. 2025
        z_array = np.linspace(0, 6, 10000)
        dz = z_array[1] - z_array[0]
        dV_Dz = get_dV_dz(z_array)
        rates = [ ]
        for i, z in enumerate(z_array):
            rate = R_gg(z, Rm_gg)
            value = rate * dV_Dz[i] * 1/(1+z)
            rates.append(value.value_in(units.yr**-1))
        Ncum = np.cumsum(rates) * dz
        plt.plot(z_array, Ncum)
        plt.show()
    
    # Likely redshifts HCSC formed when observing at z_obs
    zf_values, norm_C = sample_redshifts(
        z_min=z_obs, 
        z_max=zmax,
        Rm_gg=Rm_gg,
        Nsamples=N_zform_samp
    )

    print(f"\nFormation redshifts z [{z_obs:.2f}, {zmax:.2f}]")
    print("    Percentiles: ", np.percentile(zf_values, [25, 50, 75]))

    TDE_means = []
    GW_means  = []
    for iz, zform in enumerate(zf_values):  # For each sampled formation redshift
        print(f"\r    Sampling {iz/len(zf_values):.2%}", end="", flush=True)

        if zform > 4.0: # From Kritos et al. 2025 - negligible mergers at high z
            TDE_means.append(NULL_RATE)
            GW_means.append(NULL_RATE)
            continue
        
        tform = get_cosmic_time(zform)
        age_HCSC = t_obs - tform
        if age_HCSC < (0 | units.yr):
            print(f"Curious, HCSC formed in the future?")
            print(f"   z_obs = {z_obs}, z_form = {zform}")
            sys.exit(-1)

        TDE_local = []
        GW_local = []
        # Monte-Carlo the possible HCSC properties at z_obs (Age, kick, MBH)
        for _ in range(N_HCSC_samp):
            Mgal = sample_mass_from_PS_at_z(zform, gal_mass)
            M_bh = get_Mbh_from_Mgal(Mgal)
            v_kick = sample_kick_velocity(
                vkick_bins=Prob_Distr["Kick Lower Limit"],
                kick_probs=Prob_Distr["Hot Kick PDF"]
            )

            if v_kick < get_vesc(M_bh):
                TDE_local.append(NULL_RATE)
                GW_local.append(NULL_RATE)
                continue

            rate = gamma_tau(age_HCSC, M_bh, v_kick, gamma, f_dep=0.25, RR=RR)
            if M_bh > 1e8 | units.MSun:  # rISCO > rTidale, no TDEs
                GW_local.append(rate)
                TDE_local.append(0.0 * rate)
            else:
                TDE_local.append(TDE_FACTOR * rate)
                GW_local.append((1.0 - TDE_FACTOR) * rate)

        TDE_means.append(np.mean(TDE_local))
        GW_means.append(np.mean(GW_local))

    return norm_C * np.mean(TDE_means), norm_C * np.mean(GW_means)

def cumulative_rate(gal_mass, Rm_gg, z_grid, N_zform_samp, N_HCSC_samp, gamma=1.75, zmax=7.0, RR=0):
    """
    Compute cumulative TDE/GW event rates up to redshift z.
    Args:
        gal_mass (list):     List of galaxy masses.
        Rm_gg (function):    Interpolator for the BH-BH merger rate.
        z_grid (array):      Redshift grid considered in plotting.
        N_zform_samp (int):  Number of samples for redshift formation sampling.
        N_HCSC_samp (int):   Number of samples for Monte Carlo integration.
        gamma (float):       Power-law index for initial NSC density profile.
        zmax (float):        Maximum redshift for integration.
        RR (boolean):        Whether to calculate purely assuming RR.
    Returns:
        z_grid (array):         Redshift grid.
        TDE_cumulative (array): Cumulative TDE event rate up to z_grid.
        GW_cumulative (array):  Cumulative GW event rate up to z_grid.
    """
    dz = z_grid[1] - z_grid[0]
    TDE_cumulative = np.zeros(len(z_grid)) | units.yr**-1
    GW_cumulative  = np.zeros(len(z_grid)) | units.yr**-1
    print(f"Sampling {N_zform_samp} formation redshifts per z_obs")
    print(f"Each with {N_HCSC_samp} HCSC Monte-Carlo samples")

    ############## TO DO: PARALLELISE
    for iz, z_obs in enumerate(z_grid):
        TDE_rho, GW_rho = compute_rate_density(
            z_obs, gal_mass, 
            N_zform_samp//(iz+1),
            N_HCSC_samp,
            zmax=zmax, 
            gamma=gamma,
            Rm_gg=Rm_gg,
            RR=RR
            )

        dVc_dz = get_dV_dz(z_obs)
        time_dilation = 1.0/(1.0 + z_obs)

        dNdz_TDE = TDE_rho * dVc_dz * time_dilation
        dNdz_GW  = GW_rho * dVc_dz * time_dilation
        
        TDE_cumulative[iz] = dNdz_TDE
        GW_cumulative[iz]  = dNdz_GW

    TDE_cumulative = cumulative_trapezoid(
        TDE_cumulative.value_in(units.yr**-1), 
        z_grid, 
        initial=0
        ) | units.yr**-1
    GW_cumulative  = cumulative_trapezoid(
        GW_cumulative.value_in(units.yr**-1), 
        z_grid, 
        initial=0
        ) | units.yr**-1
    
    return TDE_cumulative, GW_cumulative

# Upper limit is 10^9 MSun BH as per Kritos    
galaxy_masses = [
    [get_Mgal_from_Mbh(1e5 | units.MSun).value_in(units.MSun), 
     get_Mgal_from_Mbh(5e5 | units.MSun).value_in(units.MSun)] | units.MSun,
    [get_Mgal_from_Mbh(1e6 | units.MSun).value_in(units.MSun), 
     get_Mgal_from_Mbh(1e8 | units.MSun).value_in(units.MSun)] | units.MSun
]

z_array = np.linspace(0.001, 3, 8)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
labels = [
    r"$10^{5} < M_{\bullet} < 5\times10^{5}$ M$_\odot$",
    r"$10^{6} < M_{\bullet} < 10^{8}$ M$_\odot$",
]
colours = ["tab:red", "tab:blue", "red", "blue"]

ALL_VELS = [[ ], [ ]]
ALL_MBHS = [[ ], [ ]]

## Idx 0: Gamma = 1.75, Mgal = 10^5-5x10^5 MSun
## Idx 1: Gamma = 1.75, Mgal = 10^6-10^9   MSun
## Idx 2: Gamma = 1.0,  Mgal = 10^5-5x10^5 MSun
## Idx 3: Gamma = 1.0,  Mgal = 10^6-10^9   MSun
z_plot = np.linspace(1e-2, 3, 2000)
fig, ax = plt.subplots(figsize=(8, 6))
ax = tickers(ax)
for ig, gamma in enumerate([1.75, 1.0]):
    print(f"Gamma = {gamma}")
    for im, gal_mass in enumerate(galaxy_masses):
        print(f"  Mgal > {gal_mass[0]}")
        if im == 0:
            Rm_interp = merger_rate_IMBH
        else:
            Rm_interp = merger_rate_SMBH

        TDEcum, GWcum = cumulative_rate(
            gal_mass, 
            Rm_interp, 
            z_array, 
            gamma=gamma,
            zmax=7.0,
            N_zform_samp=30000, 
            N_HCSC_samp=15,
            RR=False
        )

        TDE_data = PchipInterpolator(z_array, TDEcum.value_in(units.yr**-1))
        GW_data  = PchipInterpolator(z_array, GWcum.value_in(units.yr**-1))
        with open(f"plot/forecasting/MC_forecast.txt", "a") as fout:
            fout.write(f"# Gamma = {gamma}, Mgal > {gal_mass[0]}\n")
            for z in [0.1, 0.5, 1.0, 2.0, 3.0]:
                fout.write(f"z={z}, {TDE_data(z)}, {GW_data(z)}\n")
                
        ax.plot(z_plot, TDE_data(z_plot), ls='-', color=colours[ig+im*2], lw=2+im)
        if ig == 0:
            ax.plot(z_plot, GW_data(z_plot), ls=':', color=colours[ig+im*2], lw=2)

if (0):
    # Stone & Loeb 2012 rates for comparison
    TDEcum, _ = cumulative_rate(
        gal_mass, 
        Rm_interp, 
        z_array, 
        gamma=1.0,
        zmax=7.0,
        N_zform_samp=10000, 
        N_HCSC_samp=10,
        RR=True
    )
    TDE_data = PchipInterpolator(z_array, TDEcum.value_in(units.yr**-1))
    ax.plot(z_plot, TDE_data(z_plot), ls='-', color="black", lw=2+im)

ax.scatter([], [], color=colours[0], label=labels[0])
ax.scatter([], [], color=colours[2], label=labels[1])
ax.set_yscale('log')
ax.set_xlabel(r"$z_{\rm obs}$", fontsize=16)
ax.set_ylabel(r"$\dot{N}_{i}$ [yr$^{-1}$]", fontsize=16)
ax.set_xlim(6e-2, z_plot[-1])
ax.set_ylim(1e-2, ax.get_ylim()[1])
ax.legend(fontsize=12)
plt.savefig("plot/forecasting/MC_forecast.pdf", dpi=300)
plt.clf()

for i in range(2):
    x_data = np.sort(ALL_VELS[i])
    y_data = np.linspace(0, 1, len(y_data))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = tickers(ax)
    ax.plot(x_data, y_data, color='tab:blue', lw=2)
    ax.set_xlabel(r"$v_{\rm kick}$ [km s$^{-1}$]", fontsize=16)
    ax.set_ylabel(r"$f_<$", fontsize=16)
    ax.set_xlim(0, max(x_data)*1.05)
    ax.set_ylim(0, 1)
    plt.savefig(f"plot/forecasting/MC_vkick_dist_{i}.png", dpi=300)
    plt.clf()
    

for i in range(2):
    x_data = np.sort(ALL_MBHS[i])
    y_data = np.linspace(0, 1, len(y_data))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = tickers(ax)
    ax.plot(x_data, y_data, color='tab:blue', lw=2)
    ax.set_xlabel(r"$M_{\bullet}$ [M$_\odot$]", fontsize=16)
    ax.set_ylabel(r"$f_<$", fontsize=16)
    ax.set_xlim(0, max(x_data)*1.05)
    ax.set_ylim(0, 1)
    plt.savefig(f"plot/forecasting/MC_vkick_dist_{i}.png", dpi=300)
    plt.clf()