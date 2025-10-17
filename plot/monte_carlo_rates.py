import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.integrate as integ
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

from amuse.lab import units, constants


AVG_STAR_MASS = 1 | units.MSun
AVG_STAR_RAD  = 1 | units.RSun


def get_sphere_of_influence(mBH):
    """
    Get the sphere of influence radius for a black hole of mass mBH.
    Args:
        mBH (units.mass): Mass of the black hole in solar
    Returns:
        Sphere of influence radius in parsecs.
    """
    sigma = get_vdisp(mBH)
    return constants.G * mBH / sigma**2

def get_vdisp(mBH):
    """
    Get the velocity dispersion for a galaxy hosting a black hole of mass mBH.
    Uses Ferrares & Ford 2005 relation
    Args:
        mBH (units.mass): Mass of the black hole in solar masses.
    Returns:
        Velocity dispersion in km/s.
    """
    return 200 | units.kms * (mBH / (1.66e8 | units.MSun))**(1/4.86)

def get_mHCSC(mBH, vkick, gamma, rSOI=None):
    """
    Get the mass of the hypercompact stellar cluster bound to a recoiling black hole.
    Args:
        mBH (units.mass):        Mass of the black hole in solar masses.
        vkick (units.velocity):  Kick velocity in km/s.
        gamma (float):           Power-law index for the mass function.
        rSOI (units.length):     Sphere of influence radius. If None, it will be computed.
    Returns:
        Mass of the hypercompact stellar cluster in solar masses.
    """
    if rSOI is None:
        rSOI = get_sphere_of_influence(mBH)
    term = 11.6 * gamma**-1.75 * mBH * ((constants.G * mBH)/(vkick**2*rSOI))**(3-gamma)
    return term

def get_rkick(mBH, vkick):
    """
    Get the kick radius for a recoiling black hole.
    Args:
        mBH (units.mass):        Mass of the black hole in solar masses.
        vkick (units.velocity):  Kick velocity in km/s.
    Returns:
        Kick radius in parsecs.
    """
    return (8 * constants.G * mBH / vkick**2)

def get_Kepler_orbital_period(mBH, r):
    """
    Get the Keplerian orbital period at radius r around a black hole of mass mBH.
    Args:
        mBH (units.mass): Mass of the black hole in solar masses.
        r (units.length): Radius in parsecs.
    Returns:
        Orbital period in years.
    """
    return 2 * np.pi * np.sqrt(r**3 / (constants.G * mBH))

def sample_mass_from_PS_at_z(z, gmasses, ps_func):
    """
    Sample a galaxy mass from the Press-Schechter function at redshift z.
    Args:
        z (float):             Redshift at which to sample the mass.
        gmasses (units.mass):  Range of galaxy masses to sample from (AMUSE units).
        ps_func (function):    Function computing the Press-Schechter mass function.
    Returns:
        A sampled galaxy mass (in AMUSE units).
    """
    Ngrid = 100000
    
    m_min = gmasses[0].value_in(units.MSun)
    m_max = gmasses[-1].value_in(units.MSun)
    dm = m_max - m_min
    dm /= (Ngrid - 1)
    mass_grid = np.linspace(m_min, m_max, Ngrid) | units.MSun

    ps_vals = ps_func(z, mass_grid)
    ps_vals_unitless = ps_vals.value_in(units.Mpc**-3)

    cdf = np.cumsum(ps_vals_unitless) * dm
    cdf /= cdf[-1]
    
    # Create an inverse CDF interpolator.
    inv_cdf = interp1d(
        cdf, mass_grid.value_in(units.MSun), kind='linear',
        bounds_error=False, fill_value=(m_min, m_max)
    )

    random_val = np.random.uniform(0, 1)
    sampled_gal_mass = inv_cdf(random_val) | units.MSun
    return sampled_gal_mass

def sample_vkick_from_pdf(vkick_bins, pdf_vals, bin_midpoints):
    """
    Sample a vkick value from a step function PDF.
    Args:
        vkick_bins (array):     Bins for the kick velocity PDF.
        pdf_vals (function):    Function computing the PDF values at vkick_bins.
        bin_midpoints (array):  Midpoints of the vkick bins.
    Returns:
        A sampled kick velocity in kms.
    """
    bin_widths = np.diff(vkick_bins)
    bin_probs = pdf_vals(bin_midpoints) * bin_widths
    bin_probs /= bin_probs.sum()

    chosen_bin = np.random.choice(len(bin_probs), p=bin_probs)
    vkick_sample = np.random.uniform(
        vkick_bins[chosen_bin], 
        vkick_bins[chosen_bin+1]
    )
    return vkick_sample

def event_rate(
    z_range, M_range, gamma, 
    merger_rate_interp, kick_PDF,
    IMBH_IMBH_merger, N_event, 
    press_schechter, 
    num_samples
    ):
    """
    Compute the event rate using Monte Carlo integration while sampling masses
    from the Press-Schechter function.
    Args:
        z_range (float):               Redshift range (e.g. [z_min, z_max]).
        M_range (units.mass):          Galaxy mass range
        gamma (float):                 Power-law index for the mass function.
        merger_rate_interp (function): Interpolator for the merger rate.
        kick_PDF (function):           Function computing the kick velocity PDF.
        IMBH_IMBH_merger (function):   Function computing the IMBH-IMBH merger rate.
        N_event (function):            Function computing event rate.
        press_schechter (function):    Press-Schechter mass function.
        num_samples (int):             Number of Monte Carlo samples.

    Returns:
        Event rate in [yr^-1] (AMUSE units).
    """
    z_min, z_max = z_range[0], z_range[-1]
    
    TDE_vals = np.zeros(num_samples)
    GWs_vals = np.zeros(num_samples)
    for i in range(num_samples):
        z = np.random.uniform(z_min, z_max)
        
        # Sample the galaxy mass
        M_gal = sample_mass_from_PS_at_z(z, M_range, press_schechter)
        BHmass = haring_rix_relation(M_gal)
        
        # Sample the kick velocity
        v = sample_vkick_from_pdf(vkick_bins, kick_PDF, bin_midpoints) | units.kms
        v_esc = esc_velocity(haring_rix_relation(M_gal))
        if v < v_esc:# or v < (100 | units.kms):
            continue
        
        # Extract IMBH-IMBH merger rate and compute event count.
        Rm = IMBH_IMBH_merger(z, z_min, merger_rate_interp).value_in(units.yr**-1)
        event_count = N_event(M_gal, v, gamma, z)
        
        if BHmass > (1e8 | units.MSun):
            TDE_vals[i] = 0
            GWs_vals[i] = Rm * event_count
        else:
            TDE_vals[i] = TDE_FACTOR * Rm * event_count
            GWs_vals[i] = (1 - TDE_FACTOR) * Rm * event_count
    
    avg_TDEs = np.mean(TDE_vals)
    TDE_integ_estimate = avg_TDEs * (z_max - z_min)
    
    avg_GWs = np.mean(GWs_vals)
    GWs_integ_estimate = avg_GWs * (z_max - z_min)
    if TDE_integ_estimate < 0:
        print("Negative integral estimate, setting to zero")
        TDE_integ_estimate = 0
    
    return [
        TDE_integ_estimate | units.yr**-1, 
        GWs_integ_estimate | units.yr**-1
        ]

def esc_velocity(Mass):
    """
    Calculate galactic escape velocity assuming truncated isothermal sphere potential. 
    Default units in km/s.
    Args:
        Mass (units.mass): Mass of the galaxy in solar masses.
    Returns:
        Escape velocity in km/s.
    """
    vdisp = get_vdisp(Mass)
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
    rtide = (AVG_STAR_RAD.value_in(units.RSun)) * (0.844**2 * SMBH_mass/AVG_STAR_MASS)**(1./3.) | units.RSun
    rSOI = get_sphere_of_influence(SMBH_mass)
    aGW = 2*10**-4 * rSOI

    mHCSC = get_mHCSC(SMBH_mass, vkick, gamma)
    Ncluster = mHCSC / AVG_STAR_MASS
    
    rkick = get_rkick(SMBH_mass, vkick)
    Porb = get_Kepler_orbital_period(SMBH_mass, rkick)
    
    # Coeff absorbs factor of mClump, zeta (rinfl = zeta a_GW), beta for a_i vs. a_clump, k for RHill, ecc_phi for interaction time
    coeff = 2.61841299e+05
    alpha = 6.41880928e-04
    beta  = 9.55159492e+00
    
    term_a = (3-gamma) * constants.G * SMBH_mass**2 * rtide**0.5
    term_b = (8*rSOI)**(gamma-3) * vkick**-1
    term_c = aGW**(-(gamma-0.5)) * ((2*((2*gamma+3)*((2*gamma+1) - 4*gamma + 2) + 4*gamma**2 - 1))/((2*gamma - 1)*(2*gamma + 1)*(2*gamma + 3)))
    term_d = alpha/(rSOI**(3/2)/(np.sqrt(constants.G * SMBH_mass)) * (beta)**(gamma-3))
    term_e = 1/(SMBH_mass.value_in(units.MSun)**(1/3) * np.sqrt(constants.G*SMBH_mass)) * (aGW)**(3/2)
    
    Gamma0_sec = coeff * term_a * term_b * term_c * term_d * term_e / AVG_STAR_MASS * (1/1.3)

    t_exhaust = Ncluster/Gamma0_sec
    sec_decay = (rSOI**(3/2)/(np.sqrt(constants.G * SMBH_mass)) * (beta)**(gamma-3)) / alpha
    
    ### Compute timescales
    tsec   = (SMBH_mass/mHCSC) * Porb
    tRR_HA = (SMBH_mass/AVG_STAR_MASS) * Porb
    tNR_OL = (SMBH_mass/(1e5 | units.MSun))**(5/4) * (rkick/rSOI)**(1/4) | units.Gyr
    
    t_limit = min(0.5*tNR_OL, look_back(z), t_exhaust)
    sec_phase = min(t_limit, (t_exhaust)/3.)  ## See page 9, paragraph 5 Madigan
    N_sec = Gamma0_sec * sec_decay * (1-np.exp(-sec_phase/sec_decay))
    
    dM = AVG_STAR_MASS * N_sec
    mHCSC_left = mHCSC - dM
    if mHCSC_left < 0 | units.MSun:
        raise ValueError("mHCSC_left < 0")
    
    RR_phase = t_limit - sec_phase
    N_RR = 0.0
    if RR_phase > (0. | units.yr):
        CRR = 0.14
        term1 = np.log(SMBH_mass/AVG_STAR_MASS)
        term2 = np.log(rkick/rtide)
        term3 = (vkick/rkick)
        term4 = (mHCSC_left) / SMBH_mass
        
        Gamma0_RR = CRR * term1/term2 * term3 * term4
        RR_decay = 3.6 * constants.G * SMBH_mass**2 / (vkick**3 * AVG_STAR_MASS)  # From KM08
        N_RR = Gamma0_RR * RR_decay * np.exp(-sec_phase/RR_decay) * (1-np.exp(-RR_phase/RR_decay))
    if N_RR < 0:
        raise ValueError("N_RR < 0")
    if N_sec < 0:
        raise ValueError("N_sec < 0")
    return N_sec + N_RR

def IMBH_IMBH_mergers(z, z_min, merger_rate_interp):
    """
    IMBH-IMBH merger rate in [yr^-1] from arXiv:2412.15334.
    Args:
        z (float): Redshift.
        z_min (float): Minimum redshift for integration.
        merger_rate_interp (function): Interpolator for the merger rate.
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

def haring_rix_relation(mass, inverse=False):
    """
    Haring-Rix relation (BH-Galactic Bulge mass).
    Args:
        mass (units.mass): Mass of the galaxy in solar masses.
    Returns:
        Black hole mass in units.mass.
    """
    alpha_val = 8.8 + 1.24 * np.log10(mass/(1e11 | units.MSun))
    if inverse:
        return 10**((mass.value_in(units.MSun)/1e11 - 8.8)/1.24)
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

def compute_rates():
    """
    Compute event rates and plot results.
    """
    
    SMBH_masses = [4e5, 1e9] | units.MSun
    galaxy_masses = [haring_rix_relation(m, inverse=True) for m in SMBH_masses]
    
    gamma_vals = [1.75, 1.0]
    galaxy_masses = [
        np.linspace(8.63e7, 3.15e8, 100) | units.MSun,
        np.linspace(5.52e8, 1.4497406704e11, 100) | units.MSun
        #np.linspace(5.52e8, galaxy_masses[-1], 100) | units.MSun
    ]  # Upper limit is 10^9 MSun BH as per Kritos
    
    Nevents_TDE_hot  = [[ ] for _ in range(4)]
    Nevents_TDE_cold = [[ ] for _ in range(4)]
    Nevents_GW_hot   = [[ ] for _ in range(4)]
    Nevents_GW_cold  = [[ ] for _ in range(4)]
    for ig, gamma in enumerate(gamma_vals):
        for ik, vk in enumerate([Prob_Distr["Hot Kick CDF"]]):#, Prob_Distr["Cold Kick CDF"]]):
            pdf_values = np.array(vk)
            kick_PDF = interp1d(bin_midpoints, pdf_values, kind='linear', fill_value="extrapolate")
            
            for im, masses in enumerate(galaxy_masses):
                Mmax = haring_rix_relation(masses[-1]).value_in(units.MSun)
                if ik == 0:
                    print(f"...Computing for MBH < {Mmax} MSun and Hot mergers")
                else:
                    print(f"...Computing for MBH < {Mmax} MSun and Cold mergers")
                
                if Mmax < 1e6:
                    if gamma == 1.75:
                        array_idx = 0
                    elif gamma == 1.0:
                        array_idx = 2
                else:
                    if gamma == 1.75:
                        array_idx = 1
                    elif gamma == 1.0:
                        array_idx = 3

                merger_rate_interp = interp1d(merger_rate_zbins, merger_rate[im], 
                                              kind='linear', fill_value='extrapolate')
                
                N_TDEs = 0 | units.yr**-1
                N_GWs  = 0 | units.yr**-1
                for iz in range(len(redshift_bins)-1):
                    print(f"...computing z < {redshift_bins[iz+1]}...")
                    TDE_events, GW_events = event_rate(
                                z_range=[redshift_bins[iz], redshift_bins[iz+1]], 
                                M_range=masses, 
                                gamma=gamma,
                                kick_PDF=kick_PDF,
                                merger_rate_interp=merger_rate_interp,
                                IMBH_IMBH_merger=IMBH_IMBH_mergers, 
                                N_event=N_event, 
                                press_schechter=press_schechter,
                                num_samples=20000
                                )
                    N_TDEs += TDE_events
                    N_GWs  += GW_events

                    Nevents_TDE_hot[array_idx].append(N_TDEs.value_in(units.yr**-1))
                    Nevents_GW_hot[array_idx].append(N_GWs.value_in(units.yr**-1))
                        
    return Nevents_TDE_hot, Nevents_TDE_cold, Nevents_GW_hot, Nevents_GW_cold


# --- Plot and parameter definitions ---
TDE_FACTOR = 0.9
H0 = 67.4 | (units.kms/units.Mpc)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
colours = ["tab:red", "tab:blue"]
ls = ["dashed", "dashdot", "solid"]
labels = [
    r"$10^{5} < M_{\rm SMBH} < 5\times10^{5}$ M$_\odot$",
    r"$10^{6} < M_{\rm SMBH} < 10^{11}$ M$_\odot$"
]

### Probability Distributions
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

max_events = 0 | units.yr**-1

results = compute_rates()
Nevents_TDE_hot, Nevents_TDE_cold, Nevents_GW_hot, Nevents_GW_cold = results
data_df ={
    "TDE_IMBH_hot_g175":  [Nevents_TDE_hot[0], "tab:red", "-"],
    "GW_IMBH_hot_g175":   [Nevents_GW_hot[0], "tab:red", ":"],
    "TDE_SMBH_hot_g175":  [Nevents_TDE_hot[1], "red", "-",],
    "GW_SMBH_hot_g175":   [Nevents_GW_hot[1], "red", ":"],
    "TDE_IMBH_hot_g1":    [Nevents_TDE_hot[2], "tab:blue", "-"],
    "GW_IMBH_hot_g1":     [Nevents_GW_hot[2], "tab:blue", ":"],
    "TDE_SMBH_hot_g1":    [Nevents_TDE_hot[3], "blue", "-"],
    "GW_SMBH_hot_g1":     [Nevents_GW_hot[3], "blue", ":"],
}
z_range = np.linspace(0, 4, 50000)

fig, ax = plt.subplots()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.tick_params(axis="y", which='both', direction="in", labelsize=14)
ax.tick_params(axis="x", which='both', direction="in", labelsize=14)

z_record = [0.01, 0.1, 1, 2, 3, 4]
for key, value in data_df.items():
    interp_data = PchipInterpolator(redshift_bins[:-1], value[0])
    ax.plot(z_range, interp_data(z_range), color=value[1], ls=value[2], lw=3)
    for zr in z_record:
        rate_zr = interp_data(zr)
        print(f"{key} at z={zr}: {rate_zr:.2f} yr^-1")
    print("::::"*20)
ax.scatter([], [], color="tab:red", label=r"$\gamma=1.75$")
ax.scatter([], [], color="tab:blue", label=r"$\gamma=1.0$")
ax.set_xlim(1e-2, 4)
ax.set_xlabel(r"$z$", fontsize=14)
ax.set_ylabel(r"$\Gamma_{<}$ [yr$^{-1}$]", fontsize=14)
ax.set_yscale("log")
ax.legend(fontsize=13, frameon=False, loc="upper left")
ax.set_yticks([0.1, 1, 10, 100])
ax.set_ylim(0.05, 500)
ax.set_xscale("log")
plt.savefig(f"plot/figures/event_rate_vesc.pdf", bbox_inches="tight", dpi=300)
plt.clf()