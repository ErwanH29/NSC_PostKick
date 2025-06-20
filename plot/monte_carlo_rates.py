import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.integrate as integ
from scipy.interpolate import interp1d

from amuse.lab import units, constants

def sample_mass_from_PS_at_z(z, mass_range, press_schechter):
    """
    Sample a galaxy mass from the Press–Schechter function at redshift z.
    Args:
        z (float): Redshift at which to sample the mass.
        mass_range (units.mass): Range of galaxy masses to sample from (AMUSE units).
        press_schechter (function): Function computing the Press–Schechter mass function.
    Returns:
        A sampled galaxy mass (in AMUSE units).
    """
    Ngrid = 5e4
    
    m_min = mass_range[0].value_in(units.MSun)
    m_max = mass_range[-1].value_in(units.MSun)
    dm = m_max - m_min
    dm /= (Ngrid - 1)
    mass_grid = np.linspace(m_min, m_max, Ngrid) | units.MSun
    
    ps_vals = press_schechter(z, mass_grid)
    ps_vals_unitless = ps_vals.value_in(units.Mpc**-3)
    
    cdf = np.cumsum(ps_vals_unitless) * dm
    cdf /= cdf[-1]
    
    # Create an inverse CDF interpolator.
    inv_cdf = interp1d(cdf, mass_grid.value_in(units.MSun), kind='linear',
                       bounds_error=False, fill_value=(m_min, m_max))
    
    random_val = np.random.uniform(0, 1)
    sampled_gal_mass = inv_cdf(random_val) | units.MSun
    return sampled_gal_mass

def sample_vkick_from_pdf(vkick_bins, pdf_vals):
    """
    Sample a vkick value from a step function PDF.
    Args:
        vkick_bins (array): Bins for the kick velocity PDF.
        pdf_vals (function): Function computing the PDF values at vkick_bins.
    Returns:
        A sampled kick velocity in kms.
    """
    bin_widths = np.diff(vkick_bins)
    bin_probs = pdf_vals(bin_midpoints) * bin_widths
    bin_probs /= bin_probs.sum()
    
    chosen_bin = np.random.choice(len(bin_probs), p=bin_probs)
    vkick_sample = np.random.uniform(vkick_bins[chosen_bin], vkick_bins[chosen_bin+1])
    return vkick_sample

def event_rate(z_range, M_range, gamma, IMBH_IMBH_merger, N_event, press_schechter, num_samples=100):
    """
    Compute the event rate using Monte Carlo integration while sampling masses
    from the Press–Schechter function.
    Args:
        z_range (float): Redshift range (e.g. [z_min, z_max]).
        M_range (units.mass): Galaxy mass range
        kick_bins (array):  Kick velocity bins in kms
        IMBH_IMBH_merger (function): Function computing the merger rate.
        N_event (function): Function computing event rate.
        press_schechter (function): Press–Schechter mass function.
        num_samples (int): Number of Monte Carlo samples.
    
    Returns:
        Event rate in [yr^-1] (AMUSE units).
    """
    z_min, z_max = z_range[0], z_range[-1]
    
    integrand_values = np.zeros(num_samples)
    for i in range(num_samples):
        z = np.random.uniform(z_min, z_max)
        
        # Sample the galaxy mass
        M_gal = sample_mass_from_PS_at_z(z, M_range, press_schechter)
        
        # Sample the kick velocity
        v = sample_vkick_from_pdf(vkick_bins, kick_PDF) | units.kms
        v_esc = esc_velocity(haring_rix_relation(M_gal))
        if v < v_esc:
            continue
        
        # Extract IMBH-IMBH merger rate and compute event count.
        Rm = IMBH_IMBH_merger(z, z_min, None).value_in(units.yr**-1)
        event_count = N_event(M_gal, v, gamma, z)
        
        integrand_values[i] = Rm * event_count
        
    avg_integrand = np.mean(integrand_values)
    integral_estimate = avg_integrand * (z_max - z_min)
    if integral_estimate < 0:
        print("Negative integral estimate, setting to zero")
        integral_estimate = 0
    
    return integral_estimate | units.yr**-1

def esc_velocity(Mass):
    """
    Calculate galactic escape velocity assuming truncated isothermal sphere potential. 
    Default units in km/s.
    Args:
        Mass (units.mass): Mass of the galaxy in solar masses.
    Returns:
        Escape velocity in km/s.
    """
    vdisp = 200 * (Mass/(1.66*10**8 | units.MSun))**(1/4.86) | units.kms
    return 3 * vdisp

def look_back(z):
    """
    Compute look-back time in years.
    Args:
        z (float): Redshift.
    Returns:
        Look-back time in units.time.
    """
    H0 = 67.4 | (units.kms/units.Mpc)
    tH = (1/H0).value_in(units.yr)
    OmegaM = 0.303
    OmegaLambda = 0.697

    def Ez(zp):
        return np.sqrt(OmegaM * (1 + zp)**3 + OmegaLambda)

    look_back_time = integ.quad(lambda zp: tH / ((1 + zp) * Ez(zp)), 0, z)[0]
    return look_back_time | units.yr

def N_event(M, vkick, gamma, z):
    """
    Compute event rate from fit. 
    Total events assuming exhausted after 20 Myr.
    Args:
        M (units.mass): Mass of the SMBH in solar masses.
        vkick (units.velocity): Kick velocity in km/s.
        gamma (float): Power-law index for the mass function.
        z (float): Redshift.
    Returns:
        Total number of events.
    """
    SMBH_mass = haring_rix_relation(M)
    AVG_STAR_MASS = 2.43578679652 | units.MSun
    vdisp = 200 * (SMBH_mass/(1.66 * 1e8 | units.MSun))**(1/4.86) | units.kms
    rinfl = constants.G*SMBH_mass/(vdisp**2)
    rkick = 8. * constants.G*SMBH_mass/vkick**2
    rtide = (0.844**2 * SMBH_mass/AVG_STAR_MASS)**(1./3.) | units.RSun
    
    term1 = 0.14 * (SMBH_mass/AVG_STAR_MASS)**((gamma-1)/3) * (vkick/vdisp)**(-2*(gamma-1))
    term2 = np.log(SMBH_mass/AVG_STAR_MASS) / np.log(rkick/rtide)
    term3 = (vkick/rkick).value_in(units.Myr**-1)
    term4 = 11.6*gamma**-1.75 * (constants.G*SMBH_mass/(rinfl*vkick**2.))**(3.-gamma)
    
    Nrate = 31.188711107801634 * term1 * term2 * term3 * term4 | units.Myr**-1
    
    Ncluster = term4 * SMBH_mass / AVG_STAR_MASS
    time_to_exhaust = Ncluster/Nrate
    look_back_time = look_back(z)
    trelax = 10.*(SMBH_mass/(4e5 | units.MSun))**(5/4) | units.Myr
    if SMBH_mass > 1e6 | units.MSun:
        trelax = min(10*(SMBH_mass/(4e5 | units.MSun))**(5/4) | units.Myr, 300 | units.Myr)
    
    time = min(0.5*trelax, look_back_time, time_to_exhaust)
    return Nrate * time

def IMBH_IMBH_mergers(z, z_min, z_max):
    """
    IMBH-IMBH merger rate in [yr^-1] from arXiv:2412.15334.
    Args:
        z (float): Redshift.
        z_min (float): Minimum redshift for integration.
        z_max (float): Maximum redshift for integration.
    Returns:
        IMBH-IMBH merger rate in units.time
    """
    merger_rate_density = merger_rate_interp(z) | units.yr**-1 * (units.Gpc)**-3

    # Cosmological parameters
    H0 = 67.4 | (units.kms/units.Mpc)
    OmegaM = 0.303
    OmegaLambda = 0.697

    # Compute the comoving distance Dc from z_min to z.
    integrand = lambda zprime: 1.0/np.sqrt(OmegaM*(1+zprime)**3 + OmegaLambda)
    Dc = (constants.c / H0) * integ.quad(integrand, z_min, z)[0]
    Ez = np.sqrt(OmegaM*(1+z)**3 + OmegaLambda)
    
    dV_dz = 4 * np.pi * (constants.c / H0) * Dc**2 / Ez
    dNdz = merger_rate_density * dV_dz * (1+z)**-1

    return dNdz

def haring_rix_relation(mass):
    """
    Haring-Rix relation (BH-Galactic Bulge mass).
    Args:
        mass (units.mass): Mass of the galaxy in solar masses.
    Returns:
        Black hole mass in units.mass.
    """
    alpha_val = 8.8 + 1.24 * np.log10(mass/(1e11 | units.MSun))
    return 10**alpha_val | units.MSun

def press_schechter(z, M):
    """
    Compute the Press–Schechter mass function at redshift z and mass M (in Mpc^-3).
    Args:
        z (float): Redshift.
        M (units.mass): Mass in solar masses.
    Returns:
        Press–Schechter mass function value in units.length**-3.
    """
    phi_star_val = phi_star_interp(z) | (units.Mpc**-3)
    mass_param = M / (M_star_interp(z) | units.MSun)
    alpha_param = alpha_interp(z)
    dn_dM = phi_star_val * (mass_param)**(alpha_param) * np.exp(-mass_param)
    return dn_dM

# --- Plot and parameter definitions ---
TDE_FACTOR = 0.9
GW_FACTOR = 0.1
H0 = 67.4 | (units.kms/units.Mpc)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
colours = ["red", "blue"]
ls = ["dashed", "dashdot", "solid"]
labels = [
    r"$10^{5} < M_{\rm SMBH} < 5\times10^{5}$ M$_\odot$",
    r"$10^{6} < M_{\rm SMBH} < 10^{11}$ M$_\odot$"
]

Prob_Distr = {
    "Kick Lower Limit": [    0,     100,      200,       300,      400,      500,     1000,     1500,    2000],
    "Hot Kick CDF":  [0.342593, 0.211364, 0.116901, 0.078400, 0.057590, 0.140283, 0.040183, 0.010309],
    "Cold Kick CDF": [0.414482, 0.283502, 0.125030, 0.070967, 0.042490, 0.059309, 0.004030, 0.000185]
}

vkick_bins = np.array(Prob_Distr["Kick Lower Limit"]) 
bin_midpoints = (vkick_bins[:-1] + vkick_bins[1:]) / 2
V_MIN, V_MAX = vkick_bins[0], vkick_bins[-1]

redshift_bins = np.array([0.05, 0.35, 0.75, 1.5, 2.5, 3.5])
phi_star = 10**-3 * np.array([0.84, 0.84, 0.74, 0.45, 0.22, 0.12])
M_norm = 10**np.array([11.14, 11.11, 11.06, 10.91, 10.78, 10.60])
alpha_values = np.array([-1.43, -1.45, -1.48, -1.57, -1.66, -1.74])

phi_star_interp = interp1d(redshift_bins, phi_star, kind='linear', fill_value='extrapolate')
M_star_interp = interp1d(redshift_bins, M_norm, kind='linear', fill_value='extrapolate')
alpha_interp = interp1d(redshift_bins, alpha_values, kind='linear', fill_value='extrapolate')

redshift_bins = [0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]

merger_rate_zbins = [0, 0.75, 1.25, 2, 2.6, 4, 4.6]
merger_rate = [  
    [0.0006, 0.0007, 0.0037, 0.0020, 0.0055, 0.0000, 0.0025],
    [0.0005, 0.0003, 0.0025, 0.0020, 0.0045, 0.0000, 0.0000], 
]  # in yr^-1 Gpc^-3

galaxy_masses = [
    np.linspace(8.63e7, 3.15e8, 500) | units.MSun,
    np.linspace(5.52e8, 2.26380341e10, 500) | units.MSun
]
all_galaxy_masses = galaxy_masses[-1]
gamma_arr = [1.75, 1.0]

max_events = 0 | units.yr**-1

Nevents_hot =[[ ] for _ in range(4)]
Nevents_cold =[[ ] for _ in range(4)]
for i, vkick in enumerate([Prob_Distr["Hot Kick CDF"], Prob_Distr["Cold Kick CDF"]]):
    pdf_values = np.array(vkick)
    kick_PDF = interp1d(bin_midpoints, pdf_values, kind='linear', fill_value="extrapolate")
    
    for j, masses in enumerate(galaxy_masses):
        Mmax = haring_rix_relation(masses[-1]).value_in(units.MSun)
        if i == 0:
            print(f"...Computing for MSMBH < {Mmax} MSun and Hot mergers")
        else:
            print(f"...Computing for MSMBH < {Mmax} MSun and Cold mergers")
            
        merger_rate_interp = interp1d(merger_rate_zbins, merger_rate[j], 
                                      kind='linear', fill_value='extrapolate')

        Nevents = 0 | units.yr**-1
        for k in range(len(redshift_bins)-1):
            print(f"...computing z < {redshift_bins[k+1]}...")
            Nevent = event_rate(
                        z_range=[redshift_bins[k], redshift_bins[k+1]], 
                        M_range=masses, 
                        gamma=gamma_arr[0],
                        IMBH_IMBH_merger=IMBH_IMBH_mergers, 
                        N_event=N_event, 
                        press_schechter=press_schechter,
                        num_samples=20000
                        )
            Nevents += Nevent
            if i == 0:
                Nevents_hot[j].append(Nevents.value_in(units.yr**-1))
            else:
                Nevents_cold[j].append(Nevents.value_in(units.yr**-1))

for i, vkick in enumerate([Prob_Distr["Hot Kick CDF"], Prob_Distr["Cold Kick CDF"]]):
    pdf_values = np.array(vkick)
    kick_PDF = interp1d(bin_midpoints, pdf_values, kind='linear', fill_value="extrapolate")
    
    for j, masses in enumerate(galaxy_masses):
        Mmax = haring_rix_relation(masses[-1]).value_in(units.MSun)
        if i == 0:
            print(f"...Computing for MSMBH < {Mmax} MSun and Hot mergers")
        else:
            print(f"...Computing for MSMBH < {Mmax} MSun and Cold mergers")
            
        merger_rate_interp = interp1d(merger_rate_zbins, merger_rate[j], 
                                      kind='linear', fill_value='extrapolate')

        Nevents = 0 | units.yr**-1
        for k in range(len(redshift_bins)-1):
            print(f"...computing z < {redshift_bins[k+1]}...")
            Nevent = event_rate(
                        z_range=[redshift_bins[k], redshift_bins[k+1]], 
                        M_range=masses, 
                        gamma=gamma_arr[1],
                        IMBH_IMBH_merger=IMBH_IMBH_mergers, 
                        N_event=N_event, 
                        press_schechter=press_schechter,
                        num_samples=20000
                        )
            Nevents += Nevent
            if i == 0:
                Nevents_hot[j+2].append(Nevents.value_in(units.yr**-1))
            else:
                Nevents_cold[j+2].append(Nevents.value_in(units.yr**-1))
                
z_range = np.linspace(0, 4, 50000)

from scipy.interpolate import PchipInterpolator
event_SMBH_hot_g175 = PchipInterpolator(redshift_bins[:-1], Nevents_hot[1])
event_SMBH_hot_g1 = PchipInterpolator(redshift_bins[:-1], Nevents_hot[3])
event_SMBH_cold_g175 = PchipInterpolator(redshift_bins[:-1], Nevents_cold[1])
event_SMBH_cold_g1 = PchipInterpolator(redshift_bins[:-1], Nevents_cold[3])

fig, ax = plt.subplots()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.tick_params(axis="y", which='both', direction="in", labelsize=14)
ax.tick_params(axis="x", which='both', direction="in", labelsize=14)

ax.plot(z_range, TDE_FACTOR * event_SMBH_cold_g175(z_range), color="blue", label=r"$\gamma=1.75$, Cold")
ax.plot(z_range, GW_FACTOR * event_SMBH_cold_g175(z_range), color="blue", ls="-.")
ax.plot(z_range, TDE_FACTOR * event_SMBH_hot_g175(z_range), color="red", label=r"$\gamma=1.75$, Hot")
ax.plot(z_range, GW_FACTOR * event_SMBH_hot_g175(z_range), color="red", ls="-.")

ax.plot(z_range, TDE_FACTOR * event_SMBH_cold_g1(z_range), color="dodgerblue", label=r"$\gamma=1.0$, Cold")
ax.plot(z_range, GW_FACTOR * event_SMBH_cold_g1(z_range), color="dodgerblue", ls="-.")
ax.plot(z_range, TDE_FACTOR * event_SMBH_hot_g1(z_range), color="firebrick", label=r"$\gamma=1.0$, Hot")
ax.plot(z_range, GW_FACTOR * event_SMBH_hot_g1(z_range), color="firebrick", ls="-.")

ax.set_xlim(1e-2, 4)
ax.set_xlabel(r"$z$", fontsize=14)
ax.set_ylabel(r"$\Gamma_{<}$ [yr$^{-1}$]", fontsize=14)
ax.set_yscale("log")
ax.legend(fontsize=13, frameon=False, loc="upper left")
ax.set_yticks([1, 10, 100, 1000, 10000])
ax.set_ylim(0.2, 1.5 * TDE_FACTOR * np.max(event_SMBH_hot_g175(z_range)))
ax.set_xscale("log")
plt.savefig(f"plot/figures/smbh_TDE_rate_vesc.pdf", bbox_inches="tight", dpi=300)
plt.clf()


event_IMBH_hot_g175 = PchipInterpolator(redshift_bins[:-1], Nevents_hot[0])
event_IMBH_cold_g175 = PchipInterpolator(redshift_bins[:-1], Nevents_cold[0])
event_IMBH_hot_g1 = PchipInterpolator(redshift_bins[:-1], Nevents_hot[2])
event_IMBH_cold_g1 = PchipInterpolator(redshift_bins[:-1], Nevents_cold[2])

fig, ax = plt.subplots()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.tick_params(axis="y", which='both', direction="in", labelsize=14)
ax.tick_params(axis="x", which='both', direction="in", labelsize=14)

ax.plot(z_range, TDE_FACTOR * event_IMBH_cold_g175(z_range), color="blue", label=r"$\gamma=1.75$, Cold")
ax.plot(z_range, GW_FACTOR * event_IMBH_cold_g175(z_range), color="blue", ls="-.")
ax.plot(z_range, TDE_FACTOR * event_IMBH_hot_g175(z_range), color="red", label=r"$\gamma=1.75$, Hot")
ax.plot(z_range, GW_FACTOR * event_IMBH_hot_g175(z_range), color="red", ls="-.")

ax.plot(z_range, TDE_FACTOR * event_IMBH_cold_g1(z_range), color="dodgerblue", label=r"$\gamma=1.0$, Cold")
ax.plot(z_range, GW_FACTOR * event_IMBH_cold_g1(z_range), color="dodgerblue", ls="-.")
ax.plot(z_range, TDE_FACTOR * event_IMBH_hot_g1(z_range), color="firebrick", label=r"$\gamma=1.0$, Hot")
ax.plot(z_range, GW_FACTOR * event_IMBH_hot_g1(z_range), color="firebrick", ls="-.")

ax.set_xlim(1e-2, 4)
ax.set_xlabel(r"$z$", fontsize=14)
ax.set_ylabel(r"$\Gamma_{<}$ [yr$^{-1}$]", fontsize=14)
ax.set_yscale("log")
ax.set_yticks([0.1, 1, 10, 100])
ax.set_yticklabels(['0.1', '1', '10', '100'])
ax.set_ylim(0.07, 1.5 * TDE_FACTOR * np.max(event_IMBH_cold_g175(z_range)))
ax.legend(fontsize=13, frameon=False, loc="upper left")
ax.set_xscale("log")
plt.savefig(f"plot/figures/imbh_TDE_rate_vesc.pdf", bbox_inches="tight", dpi=300)