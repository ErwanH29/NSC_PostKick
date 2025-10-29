import matplotlib.pyplot as plt
import numpy as np
import sys

from amuse.lab import units

from plot.forecasting.cosmological_functions import get_cosmic_time, get_dt_dz, R_gg, get_dV_dz
from plot.forecasting.forecast_parameters import TDE_FACTOR, merger_rate_IMBH, merger_rate_SMBH, Prob_Distr
from plot.forecasting.gal_BH_relations import get_vesc, get_Mgal_from_Mbh
from plot.forecasting.HCSC_parameters import gamma_tau
from plot.forecasting.sampling_functions import sample_mass_from_PS_at_z, sample_kick_velocity, sample_redshifts


def compute_rate_density(z_obs, gal_mass, gamma, zmax, Nsamples, Rm_gg):
    """
    Compute the TDE/GW event rate up to redshift zmax. [yr^-1 Mpc^-3]
    Default zmax is set to 8.0 following Furlong et al. 2015.
    Args:
        z_obs (float):    Redshift up to which to compute the rate.
        gal_mass (list):  List of galaxy mass ranges in solar masses.
        gamma (float):    Power-law index for initian NSC density profile.
        zmax (float):     Maximum redshift for integration.
        Nsamples (int):   Number of samples for Monte Carlo integration.
        Rm_gg (function): Interpolator for the BH-BH merger rate.
    Returns: Computed event rate in yr^-1.
    """
    t_obs = get_cosmic_time(z_obs)
    
    # Possible redshift formation of HCSC at z_obs
    zf_values, z_grid = sample_redshifts(
        z_min=z_obs, 
        z_max=zmax, 
        Rm_interp=Rm_gg, 
        Nsamples=40
    )
    print(f"Sampled {len(zf_values)} formation redshifts between z={z_obs:.2f} and z={zmax:.2f}")
    print(f"z_grid: {z_grid[0]:.2f} - {z_grid[-1]:.2f}")

    TDE_results = 0.0 | units.yr**-1 / units.Mpc**3
    GW_results  = 0.0 | units.yr**-1 / units.Mpc**3
    for zform in zf_values:
        print(f"Sampling zf={zform:.2f} between z_obs={z_obs:.2f} and zmax={zmax:.2f}, gamma={gamma}", end="\r", flush=True)
        tform = get_cosmic_time(zform)
        tau = t_obs - tform  # Time delay between formation z and observed z
        if tau < 0 | units.yr:
            print("Curious case: HCSC born in the future?")
            print(f"z_obs={z_obs}, z_form={zform}, t_obs={t_obs.in_(units.Gyr)}, t_form={tform.in_(units.Gyr)}")
            print(f"z_grid={z_grid[0]:.2f}-{z_grid[-1]:.2f}")
            print(f"Error @ {zform:.2f}: tau={tau.in_(units.yr)} < 0")
            sys.exit(-1)
        
        # Monte Carlo sampling over PDFs (vkick, Mgal)
        TDE_rate_i = [ ]
        GW_rate_i  = [ ]
        for _ in range(Nsamples):  # Get a probability of HCSC states (age) at z_obs
            if zform > 4.0:  # No mergers at high z per Kritos et al. 2025
                TDE_rate_i.append(0.0 | units.yr**-1)
                GW_rate_i.append(0.0 | units.yr**-1)
            else:
                Mgal = sample_mass_from_PS_at_z(zform, mgal_lim=gal_mass)
                M_bh = get_Mgal_from_Mbh(Mgal, inverse=False)
                v_kick = sample_kick_velocity(
                    vkick_bins=Prob_Distr["Kick Lower Limit"],
                    kick_probs=Prob_Distr["Hot Kick PDF"]
                )

                v_esc = get_vesc(M_bh)
                if v_kick < v_esc:
                    TDE_rate_i.append(0.0 | units.yr**-1)
                    GW_rate_i.append(0.0 | units.yr**-1)
                else: # Get rate for the probabilistically sampled HCSC
                    rate = gamma_tau(tau, M_bh, v_kick, gamma, f_dep=0.5)
                    if M_bh > 1e8 | units.MSun:
                        GW_rate_i.append(rate)
                        TDE_rate_i.append(0. * rate)  # rISCO > rTDE
                    else:
                        TDE_rate_i.append(TDE_FACTOR * rate)
                        GW_rate_i.append((1. - TDE_FACTOR) * rate)

        # Get rates typical of HCSC observed at z_form
        TDE_mean_rate = np.mean(TDE_rate_i)
        GW_mean_rate  = np.mean(GW_rate_i)

    coeff_a = R_gg(z=z_grid, Rm_interp=Rm_gg).value_in(units.yr**-1 * units.Mpc**-3)
    coeff_b = np.abs(get_dt_dz(z_grid).value_in(units.yr))
    weights = coeff_a * coeff_b
    coeff = np.trapezoid(weights, z_grid) | units.Mpc**-3

    # Multiply typical rate with total number of HCSC formed by zform
    TDE_results = coeff * TDE_mean_rate
    GW_results  = coeff * GW_mean_rate
    return TDE_results, GW_results

def compute_rate(z_obs, gal_mass, gamma, Nsamples, Rm_gg, zmax):
    """
    Compute the TDE/GW event rate at redshift z_obs. [yr^-1]
    Args:
        z_obs (float):    Redshift at which observing.
        gal_mass (list):  List of galaxy mass ranges in solar masses.
        gamma (float):    Power-law index for initial NSC density profile.
        Nsamples (int):   Number of samples for Monte Carlo integration.
        Rm_gg (function): Interpolator for the BH-BH merger rate.
        zmax (float):     Maximum redshift for integration.
    Returns: Computed event rate in yr^-1.
    """
    TDE_rho, GW_rho = compute_rate_density(
        z_obs, gal_mass, zmax=zmax, 
        gamma=gamma, Nsamples=Nsamples, 
        Rm_gg=Rm_gg
        )
    dVc_dz = get_dV_dz(z_obs)
    time_dilation = 1.0/(1.0 + z_obs)
    return TDE_rho * dVc_dz, GW_rho * dVc_dz * time_dilation

def cumulative_rate(gal_mass, Rm_gg, z_grid, gamma=1.75, zmax=7.0, Nsamples=10000):
    """
    Compute cumulative TDE/GW event rates up to redshift z.
    Args:
        gal_mass (list):   List of galaxy masses.
        Rm_gg (function):  Interpolator for the BH-BH merger rate.
        z_grid (array):    Redshift grid considered in plotting.
        gamma (float):     Power-law index for initial NSC density profile.
        zmax (float):      Maximum redshift for integration.
        Nsamples (int):    Number of samples for Monte Carlo integration.
    Returns:
        z_grid (array):         Redshift grid.
        TDE_cumulative (array): Cumulative TDE event rate up to z_grid.
        GW_cumulative (array):  Cumulative GW event rate up to z_grid.
    """
    dz = z_grid[1] - z_grid[0]
    TDE_cumulative = np.zeros(len(z_grid)) | units.yr**-1
    GW_cumulative  = np.zeros(len(z_grid)) | units.yr**-1
    for iz, z_obs in enumerate(z_grid):  # Loop over redshift observing at
        dNdz_TDE, dNdz_GW = compute_rate(
            z_obs, gal_mass,
            gamma=gamma, 
            Nsamples=Nsamples,
            zmax=zmax,
            Rm_gg=Rm_gg
        )
        if iz == 0:
            TDE_cumulative[iz] = dNdz_TDE * dz
            GW_cumulative[iz]  = dNdz_GW  * dz
        else:
            TDE_cumulative[iz] = TDE_cumulative[iz-1] + dNdz_TDE * dz
            GW_cumulative[iz]  = GW_cumulative[iz-1]  + dNdz_GW  * dz

    return TDE_cumulative, GW_cumulative


# Upper limit is 10^9 MSun BH as per Kritos    
galaxy_masses = [
    [get_Mgal_from_Mbh(1e5 | units.MSun).value_in(units.MSun), 
     get_Mgal_from_Mbh(5e5 | units.MSun).value_in(units.MSun)] | units.MSun,
    [get_Mgal_from_Mbh(1e6 | units.MSun).value_in(units.MSun), 
     get_Mgal_from_Mbh(1e9 | units.MSun).value_in(units.MSun)] | units.MSun
]      

z_array = np.linspace(0, 3.999, 8)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
colours = ["tab:red", "tab:blue"]
labels = [
    r"$10^{5} < M_{\rm SMBH} < 5\times10^{5}$ M$_\odot$",
    r"$10^{6} < M_{\rm SMBH} < 10^{9}$ M$_\odot$",
]
colours = ["red", "blue", "tab:red", "tab:blue"]

## Idx 0: Gamma = 1.75, Mgal = 10^5-5x10^5 MSun
## Idx 1: Gamma = 1.75, Mgal = 10^6-10^9   MSun
## Idx 2: Gamma = 1.0,  Mgal = 10^5-5x10^5 MSun
## Idx 3: Gamma = 1.0,  Mgal = 10^6-10^9   MSun
fig, ax = plt.subplots(figsize=(8, 6))
for ig, gamma in enumerate([1.75, 1.0]):
    for im, gal_mass in enumerate(galaxy_masses):
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
            Nsamples=100
        )
        ax.plot(z_array, TDEcum.value_in(units.yr**-1), ls='-', color=colours[ig+im*2], label='TDEs')
        ax.plot(z_array, GWcum.value_in(units.yr**-1), ls='--', color=colours[ig+im*2], label='GWs')
ax.scatter([], [], color=colours[0], label=labels[0])
ax.scatter([], [], color=colours[1], label=labels[1])
plt.show()