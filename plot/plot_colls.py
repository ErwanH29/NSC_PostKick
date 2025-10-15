import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as mtick
import natsort
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
import sys
from matplotlib.ticker import FixedLocator, FixedFormatter

from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import read_set_from_file, units, constants
from amuse.lab import Particles


MSTAR = 1 | units.MSun
RSTAR = (MSTAR.value_in(units.MSun))**(1/3) | units.RSun
eta = 0.844
gamma = 1.75

def moving_average(array, smoothing):
    """
    Conduct running average of some variable
    
    Args:
        array (list):     Array hosting values
        smoothing (float): Number of elements to average over
    Returns:
        value (list):  List of values smoothed over some length
    """
    value = np.cumsum(array, dtype=float)
    value[smoothing:] = value[smoothing:] - value[:-smoothing]

    return value[smoothing-1:]/smoothing


class NCSCPlotter(object):
    def __init__(self):
        self.AXLABEL_SIZE = 14
        self.TICK_SIZE = 14
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        
        self.data_labels = [r"$M_{\rm SMBH} = 10^{5}$ M$_\odot$", None,
                            r"$M_{\rm SMBH} = 4 \times 10^{5}$ M$_\odot$", None]
        self.labels = [r"$300$ km s$^{-1}$", r"$600$ km s$^{-1}$"]
        self.colours = ["tab:red", "tab:blue"]
        self.colours_two = ["tab:blue", "tab:red"]
        
        cmap = matplotlib.colormaps['cool']
        self.cmap_colours = cmap(np.linspace(0.15, 1, 6))
        self.cmap_colours[2] = self.cmap_colours[1]
        self.cmap_colours[4] = self.cmap_colours[3]
        self.ls = ["-"]
        
    def extract_folder(self, SMBH_mass, vkick, folder):
        """Extract the data folders"""
        data_folders = natsort.natsorted(glob.glob(f"/media/erwanh/Expansion/All_Data/3_Runaway_BH_At_Kick/{vkick}kms_m{SMBH_mass}/Nimbh0_RA_BH_Run/{folder}/*"))
        
        return data_folders
        
    def tickers(self, ax, ptype, sig_fig):
        """
        Function to setup axis
        
        Args:
            ax (axis):  Axis needing cleaning up
            ptype (String):  Plot type (hist || plot)
            sig_fig (Int):  Number of sig. figs. on axis ticks
        Returns:
            ax (axis):  The cleaned axis
        """
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())

        if (sig_fig):
            formatter_x = StrMethodFormatter("{x:.2f}")
            formatter_y = StrMethodFormatter("{x:.0f}")
            ax.xaxis.set_major_formatter(formatter_x)
            ax.yaxis.set_major_formatter(formatter_y)

        if ptype == "hist":
            ax.tick_params(axis="y", labelsize=self.TICK_SIZE)
            ax.tick_params(axis="x", labelsize=self.TICK_SIZE)
            return ax
        else:
            ax.tick_params(axis="y", which='both', 
                           direction="in", 
                           labelsize=self.TICK_SIZE)
            ax.tick_params(axis="x", which='both', 
                           direction="in", 
                           labelsize=self.TICK_SIZE)
            return ax
    
    def ZAMS_radius(self,mass):
        log_mass = np.log10(mass.value_in(units.MSun))
        mass_sq = (mass.value_in(units.MSun))**2
        r_zams = pow(mass.value_in(units.MSun), 1.25) * (0.1148 + 0.8604*mass_sq) / (0.04651 + mass_sq)

        return r_zams | units.RSun
    
    def ecc_sma_GW(self, SMBH_mass):
        ecc_range = np.linspace(0, 0.99999, 50000)
        
        tGW = 100. | units.kyr
        mu = SMBH_mass * MSTAR
        coefficient = 256. * constants.G**3 /(5. * constants.c**5.) * (mu * (SMBH_mass + avg_star_mass)) * tGW
        sma_range = [((coefficient/(1.-i**2)**(7./2.))**(1./4.)).value_in(units.pc) for i in ecc_range]
        
        return ecc_range, sma_range
    
    def ecc_sma_tidal(self, mass, SMBH_mass):
        ecc_range = np.linspace(0, 0.99999, 50000)
        radius = self.ZAMS_radius(mass)
        rtide = radius * (0.844**2. * SMBH_mass/(mass))**(1./3.)
        sma_range = [(rtide/(1.-i)).value_in(units.pc) for i in ecc_range]
        
        return ecc_range, sma_range
    
    def filter_unbound(self, bound_particles):
        """
        Filter out unbound particles
        
        Args:
            bound_particles (Particles):  Particles to filter
        Returns:
            bound_particles (Particles):  Particles that make the HCSC
        """
        target_particle = bound_particles[bound_particles.mass.argmax()]
        minor = bound_particles - target_particle
        
        for i, p in enumerate(minor):
            sys.stdout.write(f"\rProgress: {str(100.*i/len(minor))[:5]}%")
            sys.stdout.flush()
            binary_system = Particles()
            binary_system.add_particle(target_particle)
            binary_system.add_particle(p)
            
            ke = orbital_elements(binary_system, G=constants.G)
            if ke[3] >= 1:
                bound_particles -= p
                
        return bound_particles
    
    def get_sphere_of_influence(self, mSMBH):
        """Calculate sphere of influence"""
        sigma = 200. * (mSMBH/(1.66e8 | units.MSun))**(1./4.86) | units.kms
        rSOI = constants.G * mSMBH / sigma**2.
        
        return rSOI
    
    def get_mHCSC(self, mSMBH, vkick, gamma):
        """Calculate mass of HCSC"""
        rSOI = self.get_sphere_of_influence(mSMBH)
        F1 = 11.6 * gamma **-1.75
        mHCSC = F1 * mSMBH * (constants.G * mSMBH/(rSOI * vkick**2))**(gamma-3)
        return mHCSC
    
    def plot_time_vs_coll(self):
        """Plot GW events occuring in time"""
        def custom_function(time, coeff, alpha, beta):
            rtide = RSTAR * (eta**2 * MSMBH / MSTAR)**(1/3)
            rSOI = self.get_sphere_of_influence(MSMBH)
            aGW = 2*10**-4 * rSOI
            # Coeff absorbs factor of mClump, zeta (rinfl = zeta a_GW), beta for a_i vs. a_clump, k for RHill, ecc_phi for interaction time

            term_a = (3-gamma) * constants.G * MSMBH**2 * rtide**0.5
            term_b = (8*rSOI)**(gamma-3) * VKICK**-1
            term_c = aGW**(-(gamma-0.5)) * ((2*((2*gamma+3)*((2*gamma+1) - 4*gamma + 2) + 4*gamma**2 - 1))/((2*gamma - 1)*(2*gamma + 1)*(2*gamma + 3)))
            term_d = alpha/(rSOI**(3/2)/(np.sqrt(constants.G * MSMBH)) * (beta)**(gamma-3))
            term_e = 1/(MSMBH.value_in(units.MSun)**(1/3) * np.sqrt(constants.G*MSMBH)) * (aGW)**(3/2)
            decay = np.exp(-time*term_d.value_in(units.Myr**-1))
            value = coeff * term_a * term_b * term_c * term_d * term_e / MSTAR * time * decay
            return value.value_in(units.Myr**-1)

        BIN_RESOLUTION = 2000
        TIME_PER_BIN = 50 | units.yr
        time_array = np.linspace(10**-6, 0.1, BIN_RESOLUTION)
        
        m1e5_300kms = self.extract_folder("1e5", 300, "coll_orbital")
        m1e5_600kms = self.extract_folder("1e5", 600, "coll_orbital")
        m4e5_300kms = self.extract_folder("4e5", 300, "coll_orbital")
        m4e5_600kms = self.extract_folder("4e5", 600, "coll_orbital")
        
        data_configs = [m1e5_300kms, m1e5_600kms, m4e5_300kms, m4e5_600kms]
        config_name = ["1e5_300kms", "1e5_600kms", "4e5_300kms", "4e5_600kms"]
        
        coll_keys_arr = [ ]
        coll_events_arr = [ ]
        emri_events_arr = [ ]
        tde_events_arr = [ ]
        tde_smbh_events_arr = [ ]
        ss_events_arr = [ ]
        gw_events_arr = [ ]
        N_WD_WD = 0
        for i, IC_params in enumerate(data_configs):
            coll_events_run = [ ]
            emri_events_run = [ ]
            tde_events_run = [ ]
            tde_smbh_events_run = [ ]
            ss_events_run = [ ]
            gw_events_run = [ ]
            keys = [[ ] for _ in range(len(IC_params))]
            
            NSMBH = 0
            for iter, run in enumerate(IC_params):
                data_files = natsort.natsorted(glob.glob(f"{run}/*"))
                
                if len(data_files) > 2:
                    coll_events_df = np.zeros(BIN_RESOLUTION)
                    GW_events_df = np.zeros(BIN_RESOLUTION)
                    emri_events_df = np.zeros(BIN_RESOLUTION)
                    tde_events_df = np.zeros(BIN_RESOLUTION)
                    tde_smbh_df = np.zeros(BIN_RESOLUTION)
                    ss_events_df = np.zeros(BIN_RESOLUTION)
                    
                    for file in data_files:
                        with open(file, 'rb') as df:
                            lines = [x.decode('utf8').strip() for x in df.readlines()]
                            tcoll_string = lines[0].split("Tcoll")[-1]
                            tcoll = float(tcoll_string.split(" ")[1]) | units.yr
                            key_a = int(lines[1].split("[")[-1].split("]")[0])
                            key_b = int(lines[2].split("[")[-1].split("]")[0])
                            mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                            mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun
                            type_a = int(lines[5].split("<")[1].split("- ")[0])
                            type_b = int(lines[5].split("<")[2].split("- ")[0])
                            keys[iter].append(key_a)
                            keys[iter].append(key_b)
                            if ONLY_SMBH:
                                if mass_a > 10000 | units.MSun or mass_b > 10000 | units.MSun:
                                    NSMBH += 1
                                else:
                                    continue
                            else:
                                None
                            
                            idx = int(tcoll/TIME_PER_BIN)
                            coll_events_df[idx:] += 1
                            coll_a = max(type_a, type_b)
                            coll_b = min(type_a, type_b)
                            if coll_a >= 13:
                                if coll_b >= 13:  # COMPACT - COMPACT
                                    GW_events_df[idx:] += 1
                                    if max(mass_a/mass_b, mass_b/mass_a) > 1e3:
                                        emri_events_df[idx:] += 1
                                elif coll_b > 10:  # Maguire et al. 2020
                                    if max(mass_a/mass_b, mass_b/mass_a) > 1e3:
                                        tde_events_df[idx:] += 1
                                        tde_smbh_df[idx:] += 1
                                    else:
                                        GW_events_df[idx:] += 1
                                else:
                                    tde_events_df[idx:] += 1
                                    if max(mass_a/mass_b, mass_b/mass_a) > 1e3:
                                        tde_smbh_df[idx:] += 1
                                            
                            elif coll_a >= 10: # WD
                                if coll_b >= 10:  # WD - WD
                                    GW_events_df[idx:] += 1
                                    N_WD_WD += 1
                                else:
                                    ss_events_df[idx:] += 1
                                        
                            else:
                                ss_events_df[idx:] += 1
                    coll_keys_arr.append(keys)
                    coll_events_run.append(coll_events_df)
                    emri_events_run.append(emri_events_df)
                    gw_events_run.append(GW_events_df)
                    tde_events_run.append(tde_events_df)
                    tde_smbh_events_run.append(tde_smbh_df)
                    ss_events_run.append(ss_events_df)
                    
            if len(coll_events_run) > 0:
                median_coll = np.median(coll_events_run, axis=0)
                IQRH_coll = np.percentile(coll_events_run, 75, axis=0)
                IQL_coll = np.percentile(coll_events_run, 25, axis=0)
                
                median_emri = np.median(emri_events_run, axis=0)
                IQRH_emri = np.percentile(emri_events_run, 75, axis=0)
                IQL_emri = np.percentile(emri_events_run, 25, axis=0)
                
                median_tde = np.median(tde_events_run, axis=0)
                IQRH_tde = np.percentile(tde_events_run, 75, axis=0)
                IQL_tde = np.percentile(tde_events_run, 25, axis=0)
                
                median_tde_smbh = np.median(tde_smbh_events_run, axis=0)
                IQRH_tde_smbh = np.percentile(tde_smbh_events_run, 75, axis=0)
                IQRL_tde_smbh = np.percentile(tde_smbh_events_run, 25, axis=0)
                
                median_ss = np.median(ss_events_run, axis=0)
                IQRH_ss = np.percentile(ss_events_run, 75, axis=0)
                IQL_ss = np.percentile(ss_events_run, 25, axis=0)
                
                median_gw = np.median(gw_events_run, axis=0)
                IQRH_gw = np.percentile(gw_events_run, 75, axis=0)
                IQL_gw = np.percentile(gw_events_run, 25, axis=0)
                
            coll_events_arr.append([median_coll, IQRH_coll, IQL_coll])
            emri_events_arr.append([median_emri, IQRH_emri, IQL_emri])
            gw_events_arr.append([median_gw, IQRH_gw, IQL_gw])
            tde_events_arr.append([median_tde, IQRH_tde, IQL_tde])
            tde_smbh_events_arr.append([median_tde_smbh, IQRH_tde_smbh, IQRL_tde_smbh])
            ss_events_arr.append([median_ss, IQRH_ss, IQL_ss])
        
        print(f"WD-WD collisions: {N_WD_WD/len(data_files)}")
        data_labels = [
            r"$N_{\rm coll}$", 
            r"$N_{\rm GW}$", 
            r"$N_{\rm EMRI}$", 
            r"$N_{\rm TDE}$", 
            r"$N_{\rm SMBH, TDE}$", 
            r"$N_{\rm **}$"
        ]
        
        linestyle = ["-", "-", "-.", "-", "-.", "-"]
        
        data_array = [
            coll_events_arr,
            gw_events_arr, 
            emri_events_arr, 
            tde_events_arr, 
            tde_smbh_events_arr,
            ss_events_arr
        ]
        
        x_fit = np.linspace(1e-4, 0.1, 1000)
        # Each configuration separately
        for label in range(len(config_name)):
            print(f"==== For {config_name[label]}  ====")
            
            fig, ax = plt.subplots(figsize=(6,5))
            ax.set_xlabel(r"$t$ [Myr]", fontsize=self.TICK_SIZE)
            ax.set_ylabel(r"$N_{\rm coll}$", fontsize=self.TICK_SIZE)
            self.tickers(ax, "plot", True)
            for i in range(len(data_array)-1):
                i += 1
                
                upper_smoothed = moving_average(data_array[i][label][1], 30)
                lower_smoothed = moving_average(data_array[i][label][2], 30)
                median_smoothed = moving_average(data_array[i][label][0], 30)
                time_smoothed = np.linspace(0.0, 0.1, len(upper_smoothed))
                print(f"{data_labels[i]}: final_median = {median_smoothed[-1]}, final_IQRH = {upper_smoothed[-1]}, final_IQRL = {lower_smoothed[-1]}")
                
                ax.plot(time_smoothed, upper_smoothed, 
                        color=self.cmap_colours[i],
                        alpha=0.4, lw=1)
                ax.plot(time_smoothed, lower_smoothed, 
                        color=self.cmap_colours[i],
                        alpha=0.4,lw=1)
                ax.fill_between(
                    time_smoothed, 
                    upper_smoothed,
                    lower_smoothed, 
                    color=self.cmap_colours[i], 
                    alpha=0.4
                )
                    
                ax.plot(
                    time_smoothed, median_smoothed, 
                    color=self.cmap_colours[i],
                    label=data_labels[i], zorder=1, 
                    ls=linestyle[i], lw=2
                )
                
            ax.legend(fontsize=self.TICK_SIZE, frameon=False, loc="upper left")
            ax = self.tickers(ax, "plot", True)
            #ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
            #ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
            ax.set_xlim(0, 0.1)
            ax.set_ylim(1, 130)
            #ax.set_yscale("log")
            plt.savefig(f"plot/figures/Ncoll_vs_time_{config_name[label]}.pdf", dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()
        
        fig, ax = plt.subplots(figsize=(6,5))
        ax = self.tickers(ax, "plot", True)
        ax.set_xlabel(r"$t$ [Myr]", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$N_{\rm coll}$", fontsize=self.AXLABEL_SIZE)
        for label in range(len(config_name)):
            
            if label == 0 or label == 2:
                lw = 2
                ls = "-"
            else:
                lw = 1
                ls = "-."
                    
            median_smoothed = moving_average(data_array[0][label][0], 30)
            IQRH_smoothed = moving_average(data_array[0][label][1], 30)
            IQRL_smoothed = moving_average(data_array[0][label][2], 30)
            time_smoothed = np.linspace(0.0, 0.1, len(IQRH_smoothed))
            
            if label < 2:
                MSMBH = 1e5 | units.MSun
            else:
                MSMBH = 4e5 | units.MSun
            if label%2 == 0:
                VKICK = 300 | units.kms
            else:
                VKICK = 600 | units.kms
            
            params, cov = curve_fit(
                custom_function, 
                time_smoothed,
                median_smoothed, 
                p0=[50, 0.0001, 1.0], maxfev=10000000
                )
            err = np.sqrt(np.diag(cov))
            print(f"Fit params for {MSMBH}, {VKICK}: {params}")
            print(f"Fit errors for {MSMBH}, {VKICK}: {err}")
            print(f"Nevents {MSMBH}, {VKICK} = {data_array[0][label][0][-1]}")
            print(f"===="*30)
            y_fit = custom_function(x_fit, *params)
            ax.plot(x_fit, y_fit, color="gray")
            ax.plot(time_smoothed, median_smoothed,
                    color=self.colours[label//2], lw=lw, ls=ls)
            ax.plot(time_smoothed, IQRH_smoothed, 
                    color=self.colours[label//2],
                    alpha=0.5, lw=1, ls=ls)
            ax.plot(time_smoothed, IQRL_smoothed, 
                    color=self.colours[label//2],
                    alpha=0.5,lw=1, ls=ls)
            ax.fill_between(
                time_smoothed, 
                IQRL_smoothed,
                IQRH_smoothed, 
                color=self.colours[label//2],
                alpha=0.3
            )
        ax.scatter(None, None, color="tab:red", label=r"$10^{5}$ M$_\odot$")
        ax.scatter(None, None, color="tab:blue", label=r"$4\times10^{5}$ M$_\odot$")    
        ax.legend(fontsize=self.TICK_SIZE, loc="upper left")
        #ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        #ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.set_xlim(1.e-5, 1.e-1)
        ax.set_ylim(1.e-3, 110)#data_array[0][2][1][-1])
        plt.savefig(f"plot/figures/Ncoll_vs_time_all.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
        
    def plot_coll_locs(self):
        # --- STEP 1: Gather all keys that collided with the SMBH ---
        m1e5_300kms = self.extract_folder("1e5", 300, "coll_orbital")
        m1e5_600kms = self.extract_folder("1e5", 600, "coll_orbital")
        m4e5_300kms = self.extract_folder("4e5", 300, "coll_orbital")
        m4e5_600kms = self.extract_folder("4e5", 600, "coll_orbital")

        coll_datasets = [m1e5_300kms, m1e5_600kms, m4e5_300kms, m4e5_600kms]
        smbh_collision_keys = set()

        for IC_params in coll_datasets:
            for run in IC_params:
                data_files = natsort.natsorted(glob.glob(f"{run}/*"))
                if len(data_files) < 2:
                    continue
                for file in data_files:
                    with open(file, 'rb') as df:
                        lines = [x.decode('utf8').strip() for x in df.readlines()]
                        key_a = int(lines[1].split("[")[-1].split("]")[0])
                        key_b = int(lines[2].split("[")[-1].split("]")[0])
                        mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                        mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun

                        # identify collisions involving SMBH
                        if mass_a > 10000 | units.MSun:
                            smbh_collision_keys.add(key_b)
                        elif mass_b > 10000 | units.MSun:
                            smbh_collision_keys.add(key_a)

        print(f"Collected {len(smbh_collision_keys)} SMBH-collision keys")

        # --- STEP 2: Analyze only those collisions in the final runs ---
        m1e5_300kms = self.extract_folder("1e5", 300, "simulation_snapshot")
        m1e5_600kms = self.extract_folder("1e5", 600, "simulation_snapshot")
        m4e5_300kms = self.extract_folder("4e5", 300, "simulation_snapshot")
        m4e5_600kms = self.extract_folder("4e5", 600, "simulation_snapshot")

        snapshot_datasets = [m1e5_300kms, m1e5_600kms, m4e5_300kms, m4e5_600kms]
        config_name = ["1e5_300kms", "1e5_600kms", "4e5_300kms", "4e5_600kms"]

        for ic, config in enumerate(snapshot_datasets):
            sma_col, ecc_col, aop_col, vel_col = [], [], [], []
            sma_all, aop_all, ecc_all = [], [], []

            for run in config:
                ini_dt = natsort.natsorted(glob.glob(f"{run}/*"))[0]
                fin_dt = natsort.natsorted(glob.glob(f"{run}/*"))[-1]

                p0 = read_set_from_file(ini_dt, format="amuse")
                pf = read_set_from_file(fin_dt, format="amuse")

                SMBH = p0[p0.mass.argmax()]

                # remove collided particles from final snapshot
                pf -= pf[pf.coll_events > 0]
                dp = p0 - pf  # disappeared particles

                # measure relative to SMBH reference frame
                dp.velocity -= SMBH.velocity
                dp.position -= SMBH.position

                # --- only keep those that collided with the SMBH ---
                dp_smbh = dp[[p.key in smbh_collision_keys for p in dp]]

                for particle in dp_smbh:
                    ke = orbital_elements(Particles(particles=[SMBH, particle]), G=constants.G)
                    sma_col.append(np.log10(ke[2].value_in(units.pc)))
                    ecc_col.append(ke[3])
                    aop_col.append(ke[7].value_in(units.deg))
                    
                    vis_viva = np.sqrt(constants.G * SMBH.mass * (2/particle.position.lengths() - 1/ke[2]))
                    vel_col.append(vis_viva.value_in(units.kms))
                for particle in p0:
                    if particle.mass > 10000 | units.MSun:
                        continue
                    ke = orbital_elements(Particles(particles=[SMBH, particle]), G=constants.G)
                    sma_all.append(np.log10(ke[2].value_in(units.pc)))
                    ecc_all.append(ke[3])
                    aop_all.append(ke[7].value_in(units.deg))

            mean = np.mean(vel_col)
            median = np.median(vel_col)
            iqrl = median - np.percentile(vel_col, 25)
            iqrh = np.percentile(vel_col, 75) - median
            print(f"=== Configuration: {config_name[ic]} ===")
            print(f"mean(v)={mean:.2f}, median(v)={median:.2f}, IQRLH={iqrl:.2f}-{iqrh:.2f} km/s")
            
            MSMBH_MSun = SMBH.mass.value_in(units.MSun)
            sigma = 200 * (MSMBH_MSun/(1.66*1e8))**(1./4.86) | units.kms
            rinfl = constants.G * SMBH.mass / sigma**2
            if ic % 2 == 0:
                vkick = 300 | units.kms
            else:
                vkick = 600 | units.kms
                
                
            rkick = (8 * constants.G * SMBH.mass / vkick**2).value_in(units.pc)
            ah = 2*10**-4 * rinfl.value_in(units.pc)

            data_df = {
                "ecc": [r"$e$", ecc_all, ecc_col, [0, 1]],
                "aop": [r"$\omega$ [deg]", aop_all, aop_col, [0, 360]],
            }
            
            sma_all = np.array(sma_all)
            sma_lim = [-4.5, 0.7]
            levels = [0.1, 0.5, 0.9]
            for label, data in data_df.items():
                y_data = np.array(data[1])
                values = np.vstack([sma_all, y_data])
                xx, yy = np.meshgrid(
                    np.linspace(sma_all.min(), sma_all.max(), 300),
                    np.linspace(data[3][0], data[3][1], 300)
                )
                pos = np.vstack([xx.ravel(), yy.ravel()])
                kernel = gaussian_kde(values, bw_method=0.15)
                f = np.reshape(kernel(pos).T, xx.shape)
                f_min, f_max = np.min(f), np.max(f)
                fnorm = (f - f_min) / (f_max - f_min)
            
                fig, ax = plt.subplots(figsize=(6,5))
                ax = self.tickers(ax, "hist", False)
                ax.set_xlabel(r"$a$ [pc]", fontsize=self.AXLABEL_SIZE)
                ax.set_ylabel(data[0], fontsize=self.AXLABEL_SIZE)
                cset = ax.contour(
                    xx, yy, fnorm, 
                    colors="black", 
                    levels=levels, 
                    zorder=2
                    )
                ax.clabel(cset, inline=1, fontsize=self.AXLABEL_SIZE)
                ax.contourf(
                    xx, yy, fnorm, 
                    cmap="Blues", 
                    levels=levels, 
                    zorder=1, 
                    extend="max", 
                    alpha=0.9
                    )
                ax.scatter(
                    sma_col, data[2], 
                    color="tab:red",
                    label="Colliders", 
                    zorder=3, s=5,
                ) 
                ax.set_ylim(data[3])
                ax.set_xlim(sma_lim)

                if label== "ecc":  # eccentricity plot
                    ecc_range = np.linspace(0, 0.999, 1000)
                    ah_range = ah/(1 - ecc_range)
                    ax.plot(np.log10(ah_range), ecc_range, lw=2, ls="--", color="black")
                    ax.text(
                        np.log10(ah)-0.25, 0.2, 
                        r"$a_{\rm GW}$", 
                        fontsize=self.AXLABEL_SIZE+5, 
                        rotation=85
                        )
                    ax.plot(np.log10(rkick/(1-ecc_range)), ecc_range, lw=2, ls="--", color="black")
                    ax.text(
                        np.log10(rkick)+0.03, 0.05, 
                        r"$r_{k}$",
                        fontsize=self.AXLABEL_SIZE+5, 
                        rotation=90
                        )
                floor_min_x = np.floor(np.min(sma_all))
                ceil_max_x = np.ceil(np.max(sma_all))

                # Bottom axis
                major_locs = np.arange(floor_min_x, ceil_max_x + 1)
                major_labels = [f"$10^{{{int(round(v))}}}$" for v in major_locs]

                ax.xaxis.set_major_locator(FixedLocator(major_locs))
                ax.xaxis.set_major_formatter(FixedFormatter(major_labels))
                ax.set_xticklabels(major_labels)

                minor_locs = []
                for i in major_locs[:-1]:  # avoid placing minors after max tick
                    minor_locs.extend(i + np.log10(np.arange(2, 10)))
                ax.xaxis.set_minor_locator(FixedLocator(minor_locs))
                plt.savefig(f"plot/figures/coll_locs_{label}_{config_name[ic]}.pdf", dpi=300, bbox_inches='tight')
                plt.clf()
                plt.close()     
              
vk = [500, 1000, 2000, 3000, 4000] | units.kms
masses = [1e6, 3e7] | units.MSun

factor = 0
for m in masses:
    for v in vk:
        if m > 1e7 | units.MSun:
            rinfl = 10 | units.pc
        else:
            rinfl = 3 | units.pc
        rinfl = (constants.G*m/(200*(m/(1.66*1e8 | units.MSun))**(1/4.86) | units.kms)**(2))
        factor += (200/v.value_in(units.kms))**(1-3.25) * (m.value_in(units.MSun)) ** (1-1.5)
                     
ONLY_SMBH = True
   
plot = NCSCPlotter()
#plot.plot_coll_locs()
plot.plot_time_vs_coll()