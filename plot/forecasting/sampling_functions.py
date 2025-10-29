import numpy as np
from scipy.interpolate import interp1d 
from scipy.integrate import cumulative_trapezoid
from amuse.units import units

from plot.forecasting.forecast_parameters import phi_star_interp, M_star_interp, alpha_interp
from plot.forecasting.cosmological_functions import R_gg, get_dt_dz

def sample_mass_from_PS_at_z(zf, mgal_lim, Ngrid=50000):
    """
    Sample a galaxy mass from the Press-Schechter mass function at redshift zf.
    Args:
        zf (float):        Redshift at which to sample the mass function.
        mgal_lim (float):  Mass range of the galaxy in solar units.
        Ngrid (int):       Number of grid points for numerical integration.
    Returns: Sampled galaxy mass in units.mass.
    """
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
        mass_param   = M / (M_star_interp(z) | units.MSun)
        alpha_param  = alpha_interp(z)
        dn_dM = phi_star_val * (mass_param)**(alpha_param) * np.exp(-mass_param)
        return dn_dM
    
    gal_mass_min = mgal_lim[0].value_in(units.MSun)
    gal_mass_max = mgal_lim[1].value_in(units.MSun)
    
    mass_grid = np.linspace(gal_mass_min, gal_mass_max, Ngrid) | units.MSun
    
    ps_vals = press_schechter(zf, mass_grid).value_in(units.Mpc**-3)
    cdf = np.cumsum(ps_vals)
    cdf /= cdf[-1]  # Normalize to create a proper CDF

    inv = interp1d(
        cdf, mass_grid.value_in(units.MSun), 
        kind='linear',  bounds_error=False,
        fill_value=(gal_mass_min, gal_mass_max)
    )
    return float(inv(np.random.rand())) | units.MSun

def sample_kick_velocity(vkick_bins, kick_probs):
    idx = np.random.choice(len(kick_probs), p=kick_probs)
    v0 = vkick_bins[idx]
    v1 = vkick_bins[idx + 1]
    return np.random.uniform(v0, v1) | units.kms

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

def sample_redshifts(z_min, z_max, Rm_gg, Nsamples):
    """
    Sample formation redshifts between z_min and z_max based on the merger rate density.
    Args:
        z_min (float):    Minimum redshift.
        z_max (float):    Maximum redshift.
        Rm_gg (function): Interpolator for the BH-BH merger rate.
        Nsamples (int):   Number of redshift samples to generate.
    Returns:
        z_form_samples (array): Sampled formation redshifts.
        z_grid (array):    Grid of redshifts used for sampling.
    """
    z_grid = np.linspace(z_min, z_max, Nsamples)
    R_gg_grid  = R_gg(z_grid, Rm_gg)
    weights = R_gg_grid * get_dt_dz(z_grid)
    weights = weights.value_in(units.Gpc**-3)

    cdf = cumulative_trapezoid(weights, z_grid, initial=0)
    integrand = cdf[-1]
    cdf /= integrand
    
    u = np.random.rand(Nsamples)
    z_form_samples = np.interp(u, cdf, z_grid)
    nan_mask = np.isnan(z_form_samples)
    if np.any(nan_mask):
        z_form_samples[nan_mask] = z_max
    norm_C = integrand | (units.Gpc**-3)  # Gives number of HCSC per comoving volume

    return z_form_samples, norm_C