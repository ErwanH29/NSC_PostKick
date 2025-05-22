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

from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import read_set_from_file, units, constants
from amuse.lab import Particles


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
        self.colours = ["red", "blue"]
        self.colours_two = ["blue", "red"]
        
        cmap = matplotlib.colormaps['cool']
        self.cmap_colours = cmap(np.linspace(0.15, 1, 6))
        self.cmap_colours[2] = self.cmap_colours[1]
        self.cmap_colours[4] = self.cmap_colours[3]
        self.ls = ["-"]
        
    def extract_folder(self, SMBH_mass, vkick, folder):
        """Extract the data folders"""
        data_folders = natsort.natsorted(glob.glob(f"/media/erwanh/PhD Material/All_Data/3_Runaway_BH_At_Kick/{vkick}kms_m{SMBH_mass}/Nimbh0_RA_BH_Run/{folder}/*"))
        
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
        avg_star_mass = 3.8 | units.MSun
        mu = SMBH_mass * avg_star_mass
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
    
    def stone_TDE_rate(self, mSMBH, vkick, time):
        """
        Calculate cumulative TDE rate based on equation 39 of arXiv:1105.4966 
        
        Args:
            mSMBH (mass):  SMBH mass
            vkick (float):  Kick velocity
            time (float):  Time of simulation
        Return:
            float:  Total TDE rate
        """
        rinfl = 22. * (mSMBH/(1e8 | units.MSun))**0.55  | units.pc #  Equation 14, for cusp galaxies (MSMBH < 1e8 MSun)
        rinfl = 0.23 | units.pc
        nTDE = 1.5e-6 * (mSMBH/(1e7 | units.MSun)) * (rinfl/(10. | units.pc))**-2. * (vkick/(1000. | units.kms))**-1 | units.yr**-1
        total_TDE = nTDE * time
        return total_TDE
    
    def plot_time_vs_coll(self):
        """Plot GW events occuring in time"""
        def custom_function(time, coeff):
            vdisp = 200 * (MSMBH/(1.66 * 1e8 | units.MSun))**(1/4.86) | units.kms
            rinfl = constants.G*MSMBH/(vdisp**2)
            gamma = 1.75
            rkick = 8. * constants.G*MSMBH/VKICK**2
            AVG_STAR_MASS = 2.43578679652 | units.MSun
            rtide = (0.844**2 * MSMBH/AVG_STAR_MASS)**(1./3.) | units.RSun
            
            C_RR = 0.14 * (MSMBH/AVG_STAR_MASS)**((2./3.) * (gamma-1)) * (VKICK/vdisp)**(-(4./3.) * (gamma-1)) # Try -2.67 --> \propto v^-2
            ln_term = np.log(MSMBH/AVG_STAR_MASS) / np.log(rkick/rtide)
            t_scale = VKICK/rkick
            fbound = 11.6*gamma**-1.75 * (constants.G*MSMBH/(rinfl*VKICK**2.))**(3.-gamma)
            
            term_t = C_RR * ln_term * t_scale * fbound
            term_t = term_t.value_in(units.Myr**-1)
            
            formula = (coeff) * term_t  * time
            print(f"vkick = {VKICK}, MSMBH = {MSMBH}, rinfl = {rinfl}, rkick = {rkick}, rtide = {rtide}")
            print(f"CRR={C_RR}, Ln={ln_term}, t_scale={t_scale.in_(units.Myr**-1)}, fbound={fbound}")
            print(f"In 0.1 Myr --> {coeff*C_RR*ln_term*t_scale.value_in(units.Myr**-1)*fbound*0.1}")
            print(f"Therefore coefficient {coeff} is in Myr")
            
            return formula
        
        BIN_RESOLUTION = 2000
        TIME_PER_BIN = 50 | units.yr
        time_array = np.linspace(10**-6, 0.1, BIN_RESOLUTION)
        
        m1e5_300kms = self.extract_folder("1e5", 300, "coll_orbital")
        m1e5_600kms = self.extract_folder("1e5", 600, "coll_orbital")
        m4e5_300kms = self.extract_folder("4e5", 300, "coll_orbital")
        m4e5_600kms = self.extract_folder("4e5", 600, "coll_orbital")
        
        data_configs = [m1e5_300kms, m1e5_600kms, m4e5_300kms, m4e5_600kms]
        config_name = ["1e5_300kms", "1e5_600kms", "4e5_300kms", "4e5_600kms"]
        
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
                            mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                            mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun
                            type_a = int(lines[5].split("<")[1].split("- ")[0])
                            type_b = int(lines[5].split("<")[2].split("- ")[0])
                            
                            if ONLY_SMBH:
                                if mass_a > 1000 | units.MSun or mass_b > 1000 | units.MSun:
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
            ax.set_ylim(1, 100)
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
            
            if label == 0 or label == 1:
                MSMBH = 1e5 | units.MSun
            else:
                MSMBH = 4e5 | units.MSun
            if label == 0 or label == 2:
                VKICK = 300 | units.kms
            else:
                VKICK = 600 | units.kms
            
            mask = median_smoothed != 0
            print("XXXXXXXXXXXXXXXXXXXX", median_smoothed[mask][1600])
            if MSMBH == 4e5 | units.MSun and VKICK == 300 | units.kms:
                params, covariance = curve_fit(custom_function, time_smoothed[mask][:1600], median_smoothed[mask][:1600], p0=[1], maxfev=10000)
            else:
                params, covariance = curve_fit(custom_function, time_smoothed[mask], median_smoothed[mask], p0=[1], maxfev=10000)
            print(median_smoothed[-1])
            print(f"===="*30)
            y_fit = custom_function(x_fit, *params)
            ax.plot(x_fit, y_fit, color="black", lw=0.29)
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
        ax.scatter(None, None, color="red", label=r"$10^{5}$ M$_\odot$")
        ax.scatter(None, None, color="blue", label=r"$4\times10^{5}$ M$_\odot$")    
        ax.legend(fontsize=self.TICK_SIZE, loc="upper left")
        #ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        #ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.set_xlim(1.e-5, 1.e-1)
        ax.set_ylim(1.e-3, 110)#data_array[0][2][1][-1])
        plt.savefig(f"plot/figures/Ncoll_vs_time_all.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
        
    def sma_ecc_traj_colls(self):
        """Plot collision trajectory of a specific run"""
        
        run_idx = 2
        m4e5_300kms_colls = self.extract_folder("4e5", 300, "coll_orbital")[run_idx]
        data_files = natsort.natsorted(glob.glob(f"{m4e5_300kms_colls}/*"))
        
        colliders_id_array = [ ]
        final_ecc_array = [ ]
        final_sma_array = [ ]
        for file in data_files:
            with open(file, 'rb') as df:
                lines = [x.decode('utf8').strip() for x in df.readlines()]
                key_a = int(lines[1].split("[")[-1].split("]")[0])
                key_b = int(lines[2].split("[")[-1].split("]")[0])
                mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun
                sma = float(lines[6].split(": ")[1][:-3]) | units.au
                ecc = float(lines[7].split(": ")[1])
                per = float(lines[9].split(": ")[1][:-3]) | units.deg
                
                final_sma_array.append(np.log10(sma.value_in(units.pc)))
                final_ecc_array.append(np.log10(1-ecc))
                if mass_a > mass_b:
                    colliders_id_array.append(key_b)
                else:
                    colliders_id_array.append(key_a)
        
        m4e5_300kms_snaps = self.extract_folder("4e5", 300, "simulation_snapshot")
        
        sma_array = [ ]
        ecc_array = [ ]
        coll_sma_array = {
            ID: [ ] for ID in colliders_id_array
        }
        coll_ecc_array = {
            ID: [ ] for ID in colliders_id_array
        }
        
        coll_periapse_array = {
            ID: [ ] for ID in colliders_id_array
        }
            
        for j, run in enumerate(m4e5_300kms_snaps):
            data_files = natsort.natsorted(glob.glob(f"{run}/*"))
            
            for k, dt in enumerate(data_files):
                print(f"Progress: {100*(k/len(data_files))}%", end="\r", flush=True)
                pset = read_set_from_file(dt, format="amuse")
                SMBH = pset[pset.mass.argmax()]
                minor = pset - SMBH
                
                if k == 0:
                    for p in minor:
                        bin_system = Particles(particles=[SMBH, p])
                        ke = orbital_elements(bin_system, G=constants.G)
                        if ke[3] < 1:
                            sma_array.append(np.log10(ke[2].value_in(units.pc)))
                            ecc_array.append(np.log10(1-ke[3]))
            
                if j == run_idx:
                    for ID in colliders_id_array:
                        target_minor = pset[pset.key == ID]
                        
                        if (target_minor):
                            bin_system = Particles(particles=[SMBH, target_minor])
                            ke = orbital_elements(bin_system, G=constants.G)
                            coll_sma_array[ID].append(np.log10(ke[2].value_in(units.pc)))
                            coll_ecc_array[ID].append(np.log10(1-ke[3]))
        
        levels = [1e-2, 1e-1, 0.5, 0.9]
        min_sma = -3.2
        min_ecc = -4.2
        max_sma = -1.4
        
        labs = "m4e5_300"
        smbh_masses = 4e5 | units.MSun
        
        values = np.vstack([sma_array, ecc_array])
        xx, yy = np.mgrid[min_sma:max_sma:300j, min_ecc:0:300j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kernel = gaussian_kde(values, bw_method = "silverman")
        f = np.reshape(kernel(positions).T, xx.shape)
        f_min, f_max = np.min(f), np.max(f)
        fnorm = (f - f_min) / (f_max - f_min)
        
        ecc_range_GW, sma_range_GW = self.ecc_sma_GW(smbh_masses)
        ecc_range_tidal, sma_range_tidal = self.ecc_sma_tidal(1 | units.Msun, smbh_masses)
        ecc_range_tidal, sma_range_tidal_10 = self.ecc_sma_tidal(10 | units.Msun, smbh_masses)

        
        cmap_a = matplotlib.colormaps['cool']
        cmap_a = cmap_a(np.linspace(0.1, 1, len(colliders_id_array)//2))
        cmap_b = matplotlib.colormaps['viridis']
        cmap_b = cmap_b(np.linspace(0.1, 1, len(colliders_id_array)-len(cmap_a)))
        colours = np.vstack((cmap_a, cmap_b))
        
        fig, ax = plt.subplots(figsize=(6, 6))
        self.tickers(ax, "plot", False)
        ax.set_xlim(min_sma, max_sma)
        ax.set_ylim(min_ecc, 0)
        ax.set_xlabel(r"$\log_{10}a$ [pc]", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$\log_{10}(1-e)$", fontsize=self.AXLABEL_SIZE)
        ax.contourf(xx, yy, fnorm, cmap="Blues", levels=levels, zorder=1, extend="max")
        cset = ax.contour(xx, yy, fnorm, colors="k", levels=levels, zorder=2)
        for i, ID in enumerate(colliders_id_array):
            sma = coll_sma_array[ID]
            ecc = coll_ecc_array[ID] 
            sma.append(final_sma_array[i])
            ecc.append(final_ecc_array[i])
            if np.isnan(sma[-1]) or np.isnan(ecc[-1]):
                continue
            
            ax.plot(sma, ecc, lw=0.5, color=colours[i], alpha=0.6)
            ax.scatter(sma[0], ecc[0], color=colours[i], s=20,
                       edgecolors="black", zorder=4)
            ax.scatter(final_sma_array[i], final_ecc_array[i],
                       color=colours[i], edgecolors="black", 
                       marker="X", s=30, zorder=4)
        
        ax.plot(np.log10(sma_range_GW), np.log10(1-ecc_range_GW), color="red", ls=":", zorder=1)
        ax.plot(np.log10(sma_range_tidal), np.log10(1-ecc_range_tidal), color="black", ls=":", lw=2, zorder=1)
        ax.plot(np.log10(sma_range_tidal_10), np.log10(1-ecc_range_tidal), color="black", ls="-.", lw=2, zorder=1)
        fname = f"plot/figures/coll_traj_{labs}.png"
        plt.savefig(fname, dpi=250, bbox_inches='tight')
        plt.close()
        plt.clf()

    def dperi_colls(self):
        """Plot collision trajectory of a specific run"""
        def extract_peri_coll(config):
            """Extract orbital parameters at collision"""
            coll_id_array = [ ]
            for run in config:
                data_files = natsort.natsorted(glob.glob(f"{run}/*"))
                colliders_id_df = [ ]
                
                for file in data_files:
                    with open(file, 'rb') as df:
                        lines = [x.decode('utf8').strip() for x in df.readlines()]
                        key_a = int(lines[1].split("[")[-1].split("]")[0])
                        key_b = int(lines[2].split("[")[-1].split("]")[0])
                        mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                        mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun
                        
                        if mass_a > mass_b:
                            colliders_id_df.append(key_b)
                        else:
                            colliders_id_df.append(key_a)
                
                coll_id_array.append(colliders_id_df)
                            
            return coll_id_array
        
        def create_data_frames(config, coll_id):
            """Extract orbital parameters at all steps"""
            
            sma_array = [ ]
            ecc_array = [ ]
            periapse_array = [ ]
            for id, run in enumerate(config):
                coll_sma_array = {
                    ID: [ ] for ID in coll_id[id]
                }
                coll_ecc_array = {
                    ID: [ ] for ID in coll_id[id]
                }
                coll_periapse_array = {
                    ID: [ ] for ID in coll_id[id]
                }
                
                snapshots = natsort.natsorted(glob.glob(f"{run}/*"))
                for snap in snapshots:
                    pset = read_set_from_file(snap, format="amuse")
                    SMBH = pset[pset.mass.argmax()]
                    
                    for ID in coll_id[id]:
                        target_minor = pset[pset.key == ID]
                        
                        if (target_minor):
                            bin_system = Particles(particles=[SMBH, target_minor])
                            ke = orbital_elements(bin_system, G=constants.G)
                            if abs(ke[3]) < 1:
                                coll_sma_array[ID].append(np.log10(ke[2].value_in(units.pc)))
                                coll_ecc_array[ID].append(ke[3])
                                coll_periapse_array[ID].append(ke[7].value_in(units.deg)) 
                
                sma_array.append(coll_sma_array)
                ecc_array.append(coll_ecc_array)
                periapse_array.append(coll_periapse_array)
            
            return [sma_array, ecc_array, periapse_array]
        
        def compute_dperi(periapse_array, sma_array, SMBH_mass):
            peri_array = [ ]
            for run_id, colliders in enumerate(periapse_array):
                for ID, peri_list in colliders.items():
                    if len(peri_list) < 2:
                        continue
                    
                    sma = sma_array[run_id][ID]
                    total_delta_peri = 0
                    orb_period = 2*np.pi*np.sqrt((10**sma[0] | units.pc)**3 / (constants.G * SMBH_mass))
                    for p in range(len(peri_list) - 1):
                        delta_peri = abs(peri_list[p + 1] - peri_list[p])
                        total_delta_peri += delta_peri 

                    dperi = (total_delta_peri / ((len(peri_list) - 1) * dTime)) * orb_period  # Avg. change in one orbit
                    precession_time = (2*np.pi)/dperi  # Time for one precession
                    peri_array.append(precession_time)
                    
            return peri_array
        
        def compute_dang(ecc_array, sma_array, SMBH_mass):
            rtide = (1. | units.RSun) * (0.844**2 * SMBH_mass/(1 | units.MSun))**(1/3)
            Lmin = np.sqrt(2*constants.G*SMBH_mass*rtide)
            dang_array = [ ]
            for run_id, colliders in enumerate(ecc_array):
                for id, ecc_list in colliders.items():
                    if len(ecc_list) < 2:
                        continue
                    
                    sma = sma_array[run_id][id]
                    total_delta_ang = 0
                    orb_period = 2*np.pi*np.sqrt((10**sma[0] | units.pc)**3 / (constants.G * SMBH_mass))
                    Nchanges = 0
                    for e in range(len(ecc_list) - 1):
                        if abs(ecc_list[e]) < 1 and abs(ecc_list[e + 1]) < 1:
                            Nchanges += 1
                            print(ecc_list[e], ecc_list[e + 1], sma[e], sma[e + 1])
                            ang_now = np.sqrt(constants.G * SMBH_mass * (1 - ecc_list[e]**2) * (10**sma[e] | units.pc))
                            ang_next = np.sqrt(constants.G * SMBH_mass * (1 - ecc_list[e + 1]**2) * (10**sma[e+1] | units.pc))
                            dang = abs(ang_next-ang_now)/Lmin
                            print(dang, ang_next, ang_now, Lmin)
                            total_delta_ang += dang
                    
                    dang = (total_delta_ang / (Nchanges * dTime)) * orb_period  # Avg. change in one orbit
                    ang_mom_time = 1/dang  # How many orbits to change angular momentum by Lmin
                    dang_array.append(ang_mom_time)
                    
            return dang_array
                
        m4e5_300kms_colls = self.extract_folder("4e5", 300, "coll_orbital")
        m1e5_300kms_colls = self.extract_folder("1e5", 300, "coll_orbital")
        coll_id_4e5 = extract_peri_coll(m4e5_300kms_colls)
        coll_id_1e5 = extract_peri_coll(m1e5_300kms_colls)
        
        m4e5_300kms_snaps = self.extract_folder("4e5", 300, "simulation_snapshot")
        m1e5_300kms_snaps = self.extract_folder("1e5", 300, "simulation_snapshot")
        sma_4e5, ecc_4e5, periapse_4e5 = create_data_frames(m4e5_300kms_snaps, coll_id_4e5)
        sma_1e5, ecc_1e5, periapse_1e5 = create_data_frames(m1e5_300kms_snaps, coll_id_1e5)
        
        dTime = 100 | units.yr
        dperi_4e5 = compute_dperi(ecc_4e5, sma_4e5, SMBH_mass=4e5 | units.MSun)
        dperi_1e5 = compute_dperi(ecc_1e5, sma_1e5, SMBH_mass=1e5 | units.MSun)
        dang_4e5 = compute_dang(periapse_4e5, sma_4e5, SMBH_mass=4e5 | units.MSun)
        dang_1e5 = compute_dang(periapse_1e5, sma_1e5, SMBH_mass=1e5 | units.MSun)
        
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$\log_{10}N_{\rm orb}(\delta\omega=2\pi)$", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$f_{<}$", fontsize=self.AXLABEL_SIZE)
        for i, dperi in enumerate([dperi_4e5, dperi_1e5]):
            x_sorted = np.sort(dperi)
            y_lens = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            ax.plot(np.log10(x_sorted), y_lens, lw=2, color=self.colours_two[i])
        ax.axvline(1, color="black", ls="-.")
        
        ax.scatter(None, None, color="red", label=r"$10^{5}$ M$_\odot$")
        ax.scatter(None, None, color="blue", label=r"$4\times10^{5}$ M$_\odot$")
        ax.set_ylim(0, 1.01) 
        ax.legend(fontsize=self.TICK_SIZE)
        self.tickers(ax, "plot", False)
        fname = f"plot/figures/CDF_dperi.pdf"
        plt.savefig(fname, dpi=250, bbox_inches='tight')
        plt.close()
        plt.clf()
        
        
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$\log_{10}N_{\rm orb}(\delta L=L_{\rm min})$", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$f_{<}$", fontsize=self.AXLABEL_SIZE)
        for i, dang in enumerate([dang_4e5, dang_1e5]):
            x_sorted = np.sort(dang)
            y_lens = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            ax.plot(np.log10(x_sorted), y_lens, lw=2, color=self.colours_two[i])
            ax.axhline(y_lens[abs(x_sorted - 1).argmin()], color=self.colours_two[i], ls="-.")
        
        ax.scatter(None, None, color="red", label=r"$10^{5}$ M$_\odot$")
        ax.scatter(None, None, color="blue", label=r"$4\times10^{5}$ M$_\odot$")
        ax.set_ylim(0, 1.01)
        ax.legend(fontsize=self.TICK_SIZE)
        self.tickers(ax, "plot", False)
        fname = f"plot/figures/CDF_dAng.pdf"
        plt.savefig(fname, dpi=250, bbox_inches='tight')
        plt.close()
        plt.clf()
        
    def plot_coll_traj(self):
        """Plot collision trajectory of a specific run"""
        
        run_idx = 2
        m4e5_300kms_colls = self.extract_folder("4e5", 300, "coll_orbital")[run_idx]
        data_files = natsort.natsorted(glob.glob(f"{m4e5_300kms_colls}/*"))
        
        colliders_id_array = [ ]
        for file in data_files:
            with open(file, 'rb') as df:
                lines = [x.decode('utf8').strip() for x in df.readlines()]
                key_a = int(lines[1].split("[")[-1].split("]")[0])
                key_b = int(lines[2].split("[")[-1].split("]")[0])
                mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun
                
                if mass_a > mass_b:
                    colliders_id_array.append(key_b)
                else:
                    colliders_id_array.append(key_a)
        
        m4e5_300kms_snaps = self.extract_folder("4e5", 300, "simulation_snapshot")
        print(m4e5_300kms_snaps)
        
        coll_x = {
            ID: [ ] for ID in colliders_id_array
        }
        coll_y = {
            ID: [ ] for ID in colliders_id_array
        }
        
        coll_z = {
            ID: [ ] for ID in colliders_id_array
        }
            
        for j, run in enumerate(m4e5_300kms_snaps):
            data_files = natsort.natsorted(glob.glob(f"{run}/*"))
            
            if j == run_idx:
                for k, dt in enumerate(data_files):
                    print(f"Progress: {100*(k/len(data_files))}%", end="\r", flush=True)
                    pset = read_set_from_file(dt, format="amuse")
                    SMBH = pset[pset.mass.argmax()]
                    pset.position -= SMBH.position
            
                    for ID in colliders_id_array:
                        target_minor = pset[pset.key == ID]
                        coll_x[ID] = np.concatenate((coll_x[ID], target_minor.x.value_in(units.pc)), axis=None)
                        coll_y[ID] = np.concatenate((coll_y[ID], target_minor.y.value_in(units.pc)), axis=None)
                        coll_z[ID] = np.concatenate((coll_z[ID], target_minor.z.value_in(units.pc)), axis=None)
        
        for ID in colliders_id_array:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            ax1.set_xlabel(r"$x$ [pc]", fontsize=self.AXLABEL_SIZE)
            ax1.set_ylabel(r"$y$ [pc]", fontsize=self.AXLABEL_SIZE)
            ax2.set_xlabel(r"$x$ [pc]", fontsize=self.AXLABEL_SIZE)
            ax2.set_ylabel(r"$z$ [pc]", fontsize=self.AXLABEL_SIZE)
            ax3.set_xlabel(r"$y$ [pc]", fontsize=self.AXLABEL_SIZE)
            ax3.set_ylabel(r"$z$ [pc]", fontsize=self.AXLABEL_SIZE)
            
            ax1.scatter(coll_x[ID], coll_y[ID], s=1)
            ax2.scatter(coll_x[ID], coll_z[ID], s=1)
            ax3.scatter(coll_y[ID], coll_z[ID], s=1)
            
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(-0.05, 0.05)
                ax.set_ylim(-0.05, 0.05)
            
            plt.savefig(f"plot/figures/coll_traj_{ID}.png", dpi=250, bbox_inches='tight')
            plt.close()
            plt.clf()
              
              
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
                     
ONLY_SMBH = False
   
plot = NCSCPlotter()
plot.plot_time_vs_coll()