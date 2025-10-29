import gc
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.integrate as integ
from scipy.interpolate import interp1d, PchipInterpolator
import psutil, os
from memory_profiler import profile

print(f"Memory: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")
gc.enable()

from amuse.lab import units, constants


AVG_STAR_MASS  = 1 | units.MSun
AVG_STAR_RAD   = 1 | units.RSun
TDE_FACTOR     = 0.9
SWITCH_FACTOR  = 1/3
DEPLETE_FACTOR = 0.75
H0  = 67.4 | (units.kms/units.Mpc)
TH0 = (1/H0).value_in(units.yr)
OMEGA_M = 0.303
OMEGA_L = 0.697


_ps_inv_cache = {}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
colours = ["tab:red", "tab:blue"]
labels = [
    r"$10^{5} < M_{\rm SMBH} < 5\times10^{5}$ M$_\odot$",
    r"$10^{6} < M_{\rm SMBH} < 10^{9}$ M$_\odot$",
]

################# COSMOLOGY FUNCTIONS #################
def get_Ez(z):
    return np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_L)

def get_look_back(z):
    """
    Compute look-back time in years.
    Args:
        z (float): Redshift.
    Returns:
        Look-back time in units.time.
    """
    tH = (1/H0).value_in(units.yr)
    look_back_time = integ.quad(lambda zp: tH / ((1 + zp) * get_Ez(zp)), 0, z)[0]
    return look_back_time | units.yr

def get_cosmic_time(z):
    """
    Get age of the Universe at redshift z in years.
    Args:
        z (float): Redshift.
    Returns:
        Cosmic time in units.time.
    """
    t0 = get_look_back(np.inf)
    return (t0 - get_look_back(z))

def get_comoving_distance(z):
    """
    Compute comoving distance in Mpc.
    Args:
        z (float): Redshift.
    Returns:
        Comoving distance in units.length. 
    """
    z = np.atleast_1d(z)
    integrand = lambda zprime: 1.0 / get_Ez(zprime)
    Dc_vals = np.array([
        integ.quad(integrand, 0.0, zi)[0] for zi in z
    ])
    return (constants.c / H0) * Dc_vals

def get_dV_dz(z):
    """
    Compute the comoving volume element dV/dz in Mpc^3.
    Args:
        z (float): Redshift.
    Returns:
        Comoving volume element in units.length**3.
    """
    Dc = get_comoving_distance(z)
    dV_dz = 4 * np.pi * (constants.c / H0) * Dc**2 / get_Ez(z)
    return dV_dz

################# GALACTIC-BH RELATIONS #################
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

def get_vesc(mass):
    """
    Calculate galactic escape velocity assuming truncated isothermal sphere potential. 
    Default units in km/s.
    Args:
        Mass (units.mass): Mass of the BH in solar masses.
    Returns:
        Escape velocity in km/s.
    """
    vdisp = get_vdisp(mass)
    return 3. * vdisp

def get_Mgal_from_Mbh(mass, inverse=True):
    """
    Haring-Rix relation (BH-Galactic Bulge mass).
    Args:
        mass (units.mass): Mass of the galaxy in solar masses.
        inverse (bool):    If True, get galaxy mass from BH mass; else get BH mass from galaxy mass.
    Returns:
        Black hole mass in units.mass.
    """
    alpha_val = 8.8 + 1.24 * np.log10(mass/(1e11 | units.MSun))
    if inverse:  # Get BH mass from galaxy mass
        return 10**((np.log10(mass.value_in(units.MSun))-8.8)/1.24) * 10**11 | units.MSun
    return 10**alpha_val | units.MSun

def get_sphere_of_influence(mBH):
    """
    Get the sphere of influence radius for a black hole of mass mBH.
    Args:
        mBH (units.mass): Mass of the black hole in solar masses.
    Returns:
        Sphere of influence radius in units.length.
    """
    sigma = get_vdisp(mBH)
    return (constants.G * mBH / sigma**2)

################# RECOILING CLUSTER PROPERTIES #################
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

def _HCSC_params(M_bh, vkick, gamma):
    """
    Compute event rate from fit. 
    Total events assuming exhausted after 20 Myr.
    Args:
        M_bh (units.mass):       Mass of the SMBH in solar masses.
        vkick (units.velocity):  Kick velocity in km/s.
        gamma (float):           Power-law index for the mass function.
        t_age (units.time):      Time since merger in years.
    Returns:
        Total number of events.
    """
    rtide = (AVG_STAR_RAD.value_in(units.RSun)) * (0.844**2 * M_bh/AVG_STAR_MASS)**(1./3.) | units.RSun
    rSOI  = get_sphere_of_influence(M_bh)
    aGW   = 2e-4 * rSOI
    mHCSC = get_mHCSC(M_bh, vkick, gamma, rSOI)
    rkick = get_rkick(M_bh, vkick)
    Nclst = mHCSC / AVG_STAR_MASS

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
    Gamma0_sec = coeff * term_a * term_b * term_c * term_d * term_e / AVG_STAR_MASS * (1/1.3)

    ### Compute timescales
    tau_sec = (rSOI**(3/2)/(np.sqrt(constants.G * M_bh)) * (beta)**(gamma-3)) / alpha
    #tNR_OL  = (M_bh/(1e5 | units.MSun))**(5/4) * (rkick/rSOI)**(1/4) | units.Gyr
    t_exhaust = Nclst/Gamma0_sec
    t_switch  = t_exhaust / 3.
    
    ### Resonant Relaxation Phase ###
    CRR = 0.14
    term1 = np.log(M_bh/AVG_STAR_MASS)
    term2 = np.log(rkick/rtide)
    term3 = (vkick/rkick)
    term4 = (mHCSC * SWITCH_FACTOR) / M_bh
    tau_RR = 3.6 * constants.G * M_bh**2 / (vkick**3 * AVG_STAR_MASS)  # From KM08
    Gamma0_RR = CRR * term1/term2 * term3 * term4

    return Gamma0_sec, tau_sec, Gamma0_RR, tau_RR, t_switch, mHCSC, Nclst

################# SAMPLING FUNCTIONS #################
def sample_ps_func(z, M):
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
    return phi_star_val * (mass_param)**(alpha_param) * np.exp(-mass_param)

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
    z_key = round(float(z), 1)
    inv = _ps_inv_cache.get(z_key)
    if inv is None:
        Ngrid = 50000
        
        m_min = gmasses[0].value_in(units.MSun)
        m_max = gmasses[-1].value_in(units.MSun)
        mass_grid = np.linspace(m_min, m_max, Ngrid) | units.MSun

        # Create an inverse CDF interpolator.
        ps_vals = ps_func(z, mass_grid).value_in(units.Mpc**-3)
        cdf = np.cumsum(ps_vals)
        cdf /= cdf[-1]
        inv = interp1d(
            cdf, mass_grid.value_in(units.MSun), kind='linear',
            bounds_error=False, fill_value=(m_min, m_max)
        )
        _ps_inv_cache[z_key] = inv

    return (inv(np.random.rand()) | units.MSun)

def sample_vkick_from_bins(vkick_bins, kick_probs):
    idx = np.random.choice(len(kick_probs), p=kick_probs)
    v0, v1 = vkick_bins[idx], vkick_bins[idx+1]
    return (np.random.uniform(v0, v1)) | units.kms

def sample_BBH_merger(z, Rm_interp):
    """
    IMBH-IMBH merger rate in [yr^-1] from arXiv:2412.15334.
    Args:
        z (float): Redshift.
        Rm_interp (function): Interpolator for the merger rate.
    Returns:
        IMBH-IMBH merger rate in units.time
    """
    merger_rate_density = Rm_interp(z) | units.yr**-1 * (units.Gpc)**-3
    return merger_rate_density

def sample_redshifts(z_min, z_max, Rm_interp, Nsamples=10000):
    z_grid = np.linspace(z_min, z_max, Nsamples)
    w  = Rm_interp(z_grid) * TH0 / ((1.0 + z_grid) * get_Ez(z_grid))
    W  = np.cumsum(w)
    if W[-1] <= 0:
        sys.exit(-1)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    
    u = np.random.rand(Nsamples)
    z_form_samples = np.interp(u, cdf, z_grid)

    return z_form_samples, z_grid

################## EVENT RATE CALCULATIONS #################
def scale_Nform(z_emit, Zmax_form,
                N_base=5000, alpha=0.75,
                Nmin=200, Nmax=5000):
    """
    Scale the number of formation redshift samples based on the merger integral.
    Args:
        z_emit (float):       Emission redshift.
        Zmax_form (float):    Maximum formation redshift.
        I_emit (units.length**-3): Merger integral at z_emit.
        I_ref (units.length**-3):  Reference merger integral.
        dz_ref (float):       Reference redshift bin width.
        N_base (int):         Base number of samples.
        alpha (float):        Exponent for I_emit scaling.
        beta (float):         Exponent for dz scaling.
        Nmin (int):           Minimum number of samples.
        Nmax (int):           Maximum number of samples.
    Returns:
        Scaled number of formation redshift samples (int).
    """
    dz_form = max(Zmax_form - z_emit, 1e-3)
    scale_dz = (dz_form / (Zmax_form))**alpha
    N = int(np.clip(N_base * scale_dz, Nmin, Nmax))
    return N

def inst_rate_at_time(Gamma0_sec, tau_sec, Gamma0_RR, tau_RR, t_switch, t_age):
    """
    Compute the instantaneous event rate at time t_age.
    Args:
        M_bh (units.mass):       Mass of the SMBH in solar masses.
        vkick (units.velocity):  Kick velocity in km/s.
        gamma (float):           Power-law index for the mass function.
        t_age (units.time):      Time since merger in years.
    Returns:
        Instantaneous event rate in yr^-1.
    """
    if t_age <= t_switch:
        rate = Gamma0_sec * np.exp(-t_age / tau_sec)
    else:
        rate = Gamma0_RR * np.exp(-(t_age - t_switch) / tau_RR)
    return rate

def active_HCSC_lifetime(M_bh, vkick, gamma, f_dep):
    """
    Compute the active lifetime of the HCSC given the parameters.
    Args:
        M_bh (units.mass):       Mass of the SMBH in solar masses.
        vkick (units.velocity):  Kick velocity in km/s.
        gamma (float):           Power-law index for the mass function.
        f_dep (float):           Depletion factor for the cluster.
    Returns:
        Active lifetime in years.
    """
    Gamma0_sec, tau_sec, Gamma0_RR, tau_RR, t_switch, mHCSC, Nclst = _HCSC_params(M_bh, vkick, gamma)
    N_target = f_dep * Nclst
    
    N_burst_at_switch = Gamma0_sec * tau_sec * (1.0 - np.exp(-t_switch / tau_sec))
    if N_target <= N_burst_at_switch: # Exhausted in burst phase
        t_life = -tau_sec * np.log(1 - N_target / (Gamma0_sec * tau_sec))
        return Gamma0_sec, tau_sec, Gamma0_RR, tau_RR, t_switch, t_life

    N_post_burst = N_target - N_burst_at_switch
    N_RR_max = Gamma0_RR * tau_RR
    if N_post_burst >= N_RR_max:
        t_life = np.inf | units.yr  # Never exhausts
        return Gamma0_sec, tau_sec, Gamma0_RR, tau_RR, t_switch, t_life
    
    t_life = t_switch - tau_RR * np.log(1 - N_post_burst / (Gamma0_RR * tau_RR))
    return Gamma0_sec, tau_sec, Gamma0_RR, tau_RR, t_switch, t_life

def event_rate_kernel(
    z_emit_min,
    z_emit_max,
    M_range,
    gamma,
    Rm_interp,
    sample_ps_func,
    vkick_bins,
    kick_probs,
    Zmax_form=10.0,
    N_Mv=10,
    ):
    """
    Compute observed event-rate contribution from specified redshift shell.
    Args:
        z_emit_min (float):     Minimum emission redshift.
        z_emit_max (float):     Maximum emission redshift.
        M_range (units.mass):   Range of galaxy masses to sample from
        gamma (float):          Power-law index for the mass function.
        Rm_interp (function):   Interpolator for the BH-BH merger rate.
        sample_ps_func (func):  Function to sample galaxy masses from Press-Schechter.
        vkick_bins (array):     Bins for the kick velocity PDF.
        vkick_PDF (function):   Function to compute the kick velocity PDF.
        Zmax_form (float):      Maximum formation redshift.
        N_form (int):           Number of z_form samples per emission shell.
        N_Mv (int):             Number of (M,v) samples per (z_emit, z_form).
    """
    # Get the redshift at emission
    z_emit = (z_emit_min + z_emit_max) / 2
    t_emit = get_cosmic_time(z_emit)
    def integrand_form(zf):
        return (Rm_interp(zf) * TH0 / ((1+zf) * get_Ez(zf)))

    I_emit = integ.quad(integrand_form, z_emit, Zmax_form, limit=200)[0] | units.Gpc**-3
    if I_emit <= (0 | units.Gpc**-3):
        return (0 | units.yr**-1), (0 | units.yr**-1)
    
    N_form = scale_Nform(z_emit, Zmax_form)
    z_form_samples, zf_grid = sample_redshifts(
        z_emit, Zmax_form, Rm_interp, Nsamples=N_form
        )
    
    dV_dz_emit = get_dV_dz(z_emit)
    dz_obs = (z_emit_max - z_emit_min) / (1 + z_emit)

    avg_TDE_rate = 0 | units.yr**-1
    avg_GW_rate  = 0 | units.yr**-1
    for iz, z_form in enumerate(z_form_samples):
        if iz%200==0:
            gc.collect()

        t_form = get_cosmic_time(z_form)
        t_age  = t_emit - t_form  # Age of HCSC at observation time
        if t_age <= (0 | units.yr):
            continue
        print(
            f"\r    Sampling {z_form:.3f}. \
                    Progress {iz+1}/{N_form}, \
                    Memory={psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB", 
                    end="", flush=True
            )
        
        mc_TDE = 0 | units.yr**-1
        mc_GW  = 0 | units.yr**-1
        for _ in range(N_Mv):
            # Sample galaxy mass from Press-Schechter
            M_gal = sample_mass_from_PS_at_z(z_form, M_range, sample_ps_func)
            
            # Get corresponding BH mass
            M_bh = get_Mgal_from_Mbh(M_gal, inverse=False)
            # Sample kick velocity from PDF
            vkick = sample_vkick_from_bins(vkick_bins, kick_probs)
            if vkick <= get_vesc(M_bh):
                continue  # BH retained, no recoil cluster formed

            # Compute active lifetime and event rates
            Gamma0_sec, tau_sec, Gamma0_RR, tau_RR, t_switch, t_life = active_HCSC_lifetime(
                M_bh, vkick, gamma, DEPLETE_FACTOR
            )
            if np.isfinite(t_age.number) and (t_age >= t_life):
                continue  # HCSC inactive at observation time

            rate_at_tage = inst_rate_at_time(
                Gamma0_sec, tau_sec, Gamma0_RR, tau_RR, t_switch, t_age
            )
            if M_bh < (1e8 | units.MSun):
                mc_TDE += rate_at_tage * TDE_FACTOR
                mc_GW  += rate_at_tage * (1 - TDE_FACTOR)
            else:
                mc_GW  += rate_at_tage
        
        if N_Mv > 0:
            mc_TDE /= N_Mv
            mc_GW  /= N_Mv
        
        avg_TDE_rate += mc_TDE
        avg_GW_rate  += mc_GW
        
    avg_TDE_rate /= N_form
    avg_GW_rate  /= N_form

    rho_TDE_emit = I_emit * avg_TDE_rate    # [yr^-1 Gpc^-3]
    rho_GW_emit  = I_emit * avg_GW_rate
    
    dGamma_TDE = rho_TDE_emit * dV_dz_emit * dz_obs
    dGamma_GW  = rho_GW_emit  * dV_dz_emit * dz_obs

    return dGamma_TDE, dGamma_GW

def compute_rates():
    """Compute event rates and plot results."""
    ### Probability Distributions -- Table VIII: https://arxiv.org/pdf/1201.1923
    Prob_Distr = {
        "Kick Lower Limit": [    0,     100,      200,       300,      400,      500,     1000,     1500,    2000],
        "Hot Kick PDF":  [0.342593, 0.211364, 0.116901, 0.078400, 0.057590, 0.140283, 0.040183, 0.010309],
        "Cold Kick CDF": [0.414482, 0.283502, 0.125030, 0.070967, 0.042490, 0.059309, 0.004030, 0.000185]
    }
    vkick_bins      = np.array(Prob_Distr["Kick Lower Limit"], dtype=float)
    kick_bin_probs  = np.array(Prob_Distr["Hot Kick PDF"], dtype=float)
    kick_bin_probs /= kick_bin_probs.sum()
        
    ### Press-Schechter function parameters -- Table A1: https://arxiv.org/pdf/1410.3485
    global phi_star_interp, M_star_interp, alpha_interp
    redshift_bins = np.array([0.05, 0.35, 0.75, 1.5, 2.5, 3.5])
    phi_star      = 10**-3 * np.array([0.84, 0.84, 0.74, 0.45, 0.22, 0.12])
    M_norm        = 10**np.array([11.14, 11.11, 11.06, 10.91, 10.78, 10.60])
    alpha_values  = np.array([-1.43, -1.45, -1.48, -1.57, -1.66, -1.74])
    phi_star_interp = interp1d(redshift_bins, phi_star, kind='linear', fill_value='extrapolate')
    M_star_interp   = interp1d(redshift_bins, M_norm, kind='linear', fill_value='extrapolate')
    alpha_interp    = interp1d(redshift_bins, alpha_values, kind='linear', fill_value='extrapolate')
    
    ### Binary BH merger rates -- Fig. 10: https://arxiv.org/pdf/2412.15334
    merger_rate_zbins = [0, 0.75, 1.25, 2, 2.6, 3.5, 4.0]
    merger_rate = [  
        [0.0006, 0.0007, 0.0037, 0.0020, 0.0055, 0.002,  0.],
        [0.0007, 0.0004, 0.0027, 0.0020, 0.0050, 0.,     0.], 
    ]  # in yr^-1 Gpc^-3

    merger_interp_IM = interp1d(merger_rate_zbins, merger_rate[0], kind='linear', fill_value='extrapolate')
    merger_interp_SM = interp1d(merger_rate_zbins, merger_rate[1], kind='linear', fill_value='extrapolate')

    results = {
        "TDE_IMBH_hot_g175": [], "GW_IMBH_hot_g175": [],
        "TDE_SMBH_hot_g175": [], "GW_SMBH_hot_g175": [],
        "TDE_IMBH_hot_g1":   [], "GW_IMBH_hot_g1":   [],
        "TDE_SMBH_hot_g1":   [], "GW_SMBH_hot_g1":   [],
    }

    z_bins_obs = [0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
    gamma_vals = [1.75, 1.0]
    galaxy_masses = [
        np.linspace(
            get_Mgal_from_Mbh(1e5 | units.MSun).value_in(units.MSun), 
            get_Mgal_from_Mbh(5e5 | units.MSun).value_in(units.MSun), 
            100) | units.MSun,
        np.linspace(
            get_Mgal_from_Mbh(1e6 | units.MSun).value_in(units.MSun), 
            get_Mgal_from_Mbh(1e9 | units.MSun).value_in(units.MSun), 
            100) | units.MSun
    ]  # Upper limit is 10^9 MSun BH as per Kritos
    # Loop over gamma and mass ranges
    for ig, gamma in enumerate(gamma_vals):
        for im, gal_dm in enumerate(galaxy_masses):
            print(f"Computing rates for gamma={gamma}, mass range {labels[im]}")
            # Choose merger-rate interp per mass regime (your original two curves)
            if im == 0:
                merger_interp = merger_interp_IM
            else:
                merger_interp = merger_interp_SM

            # cumulative arrays
            cum_TDE = 0 | units.yr**-1
            cum_GW  = 0 | units.yr**-1

            # Build cumulative rates
            for iz in range(len(z_bins_obs)-1):
                print(f"  At z={redshift_bins[iz]:.2f}...")
                zmin, zmax = z_bins_obs[iz], z_bins_obs[iz+1]
                TDE_shell, GW_shell = event_rate_kernel(
                    z_emit_min=zmin,
                    z_emit_max=zmax,
                    M_range=gal_dm,
                    gamma=gamma,
                    Rm_interp=merger_interp,
                    sample_ps_func=sample_ps_func,
                    vkick_bins=vkick_bins,
                    kick_probs=kick_bin_probs,
                    Zmax_form=5.0,
                    N_Mv=10
                )
                cum_TDE += TDE_shell
                cum_GW  += GW_shell
                
                print(f"\n    Cumulative TDE rate: {cum_TDE.value_in(units.yr**-1)} yr^-1")
                print(f"    Cumulative GW  rate: {cum_GW.value_in(units.yr**-1)} yr^-1")

                # Map to results keys
                if im == 0 and ig == 0:
                    results["TDE_IMBH_hot_g175"].append(cum_TDE.value_in(units.yr**-1))
                    results["GW_IMBH_hot_g175"].append(cum_GW.value_in(units.yr**-1))
                elif im == 1 and ig == 0:
                    results["TDE_SMBH_hot_g175"].append(cum_TDE.value_in(units.yr**-1))
                    results["GW_SMBH_hot_g175"].append(cum_GW.value_in(units.yr**-1))
                elif im == 0 and ig == 1:
                    results["TDE_IMBH_hot_g1"].append(cum_TDE.value_in(units.yr**-1))
                    results["GW_IMBH_hot_g1"].append(cum_GW.value_in(units.yr**-1))
                else:
                    results["TDE_SMBH_hot_g1"].append(cum_TDE.value_in(units.yr**-1))
                    results["GW_SMBH_hot_g1"].append(cum_GW.value_in(units.yr**-1))

    return z_bins_obs, results

z_bins, results = compute_rates()
series = {
    "TDE_IMBH_hot_g175":  ("tab:red",  "-", r"TDE, IMBH, $\gamma=1.75$"),
    "GW_IMBH_hot_g175":   ("tab:red",  ":", r"GW, IMBH, $\gamma=1.75$"),
    "TDE_SMBH_hot_g175":  ("red",      "-", r"TDE, SMBH, $\gamma=1.75$"),
    "GW_SMBH_hot_g175":   ("red",      ":", r"GW, SMBH, $\gamma=1.75$"),
    "TDE_IMBH_hot_g1":    ("tab:blue", "-", r"TDE, IMBH, $\gamma=1.0$"),
    "GW_IMBH_hot_g1":     ("tab:blue", ":", r"GW, IMBH, $\gamma=1.0$"),
    "TDE_SMBH_hot_g1":    ("blue",     "-", r"TDE, SMBH, $\gamma=1.0$"),
    "GW_SMBH_hot_g1":     ("blue",     ":", r"GW, SMBH, $\gamma=1.0$"),
}
z_range = np.linspace(0, 4, 50000)
z_centers = z_bins[1:]

fig, ax = plt.subplots()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
ax.tick_params(axis="y", which='both', direction="in", labelsize=14)
ax.tick_params(axis="x", which='both', direction="in", labelsize=14)

z_record = [0.01, 0.1, 1, 2, 3, 4]
for key, (c, ls, lab) in series.items():
    y = results[key]
    x = z_centers[:len(y)]
    # smooth interpolate for curve
    interp = PchipInterpolator(x, y)
    z_plot = np.linspace(1e-2, 4, 2000)
    ax.plot(z_plot, interp(z_plot), color=c, ls=ls, lw=3)
ax.scatter([], [], color="tab:red", label=r"$\gamma=1.75$")
ax.scatter([], [], color="tab:blue", label=r"$\gamma=1.0$")
ax.set_xlim(1e-2, 4)
ax.set_xlabel(r"$z$", fontsize=14)
ax.set_ylabel(r"$\Gamma_{<}$ [yr$^{-1}$]", fontsize=14)
ax.set_yscale("log")
ax.legend(fontsize=13, frameon=False, loc="upper left")
ax.set_ylim(8e-4, 22)
plt.savefig(f"plot/figures/event_rate_kernel_v>100.pdf", bbox_inches="tight", dpi=300)
plt.clf()