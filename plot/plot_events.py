import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as mticker
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
        #data_folders = natsort.natsorted(glob.glob(f"/media/erwanh/Elements/BH_Post_Kick/data/{vkick}kms_m{SMBH_mass}/Nimbh0_RA_BH_Run/{folder}/*"))
        data_folders = natsort.natsorted(glob.glob(f"data/{vkick}kms_m{SMBH_mass}/Nimbh0_RA_BH_Run/{folder}/*"))
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
            formatter = StrMethodFormatter("{x:.1f}")
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

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
        0.08353 + 0.0565*log_mass
        0.01291 + 0.2226*log_mass
        0.1151 + 0.06267*log_mass
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
    
    def plot_GW_vs_time(self):
        """Plot GW events occuring in time"""
        
        def custom_function(time, a):
            rtide = (2 | units.RSun) * (0.844**2 * MSMBH/(2 | units.MSun))**(1/3)
            R = (8.*constants.G*MSMBH)/(VKICK**2) / rtide
            rinfl = constants.G*MSMBH/(200 | units.kms * (MSMBH/(1.66 * 1e8 | units.MSun))**(1/4.86))**2
            fbound = 11.6*(1.75)**-1.75 * ((constants.G*MSMBH)/(rinfl*VKICK**2))**(3-1.75)
            formula = a/(2*np.pi*np.log(R)) * (VKICK**3/(constants.G*MSMBH)) * fbound**2 * (time | units.kyr)
            tau = (3.6 * constants.G * MSMBH**2/(VKICK**3 * (2 | units.MSun))).value_in(units.kyr)
            return formula
        
        BIN_RESOLUTION = 2000
        TIME_PER_BIN = 50 | units.yr
        time_array = np.linspace(10**-4, 100, BIN_RESOLUTION)
        
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
        for i, IC_params in enumerate(data_configs):
            coll_events_run = [ ]
            emri_events_run = [ ]
            tde_events_run = [ ]
            tde_smbh_events_run = [ ]
            ss_events_run = [ ]
            gw_events_run = [ ]
            
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
                            
                            idx = int(tcoll/TIME_PER_BIN)
                            coll_events_df[idx:] += 1
                            if type_a >= 10:
                                if type_b >= 10:  # COMPACT - COMPACT
                                    GW_events_df[idx:] += 1
                                    if max(mass_a/mass_b, mass_b/mass_a) > 1e4:
                                        emri_events_df[idx:] += 1
                                    
                                else:  # COMPACT - STAR
                                    tde_events_df[idx:] += 1
                                    if mass_a > (1e4 | units.MSun):
                                        tde_smbh_df[idx:] += 1
                            
                            else:
                                if type_b >= 10:  # STAR - COMPACT
                                    tde_events_df[idx:] += 1
                                    if mass_b > (1e4 | units.MSun):
                                        tde_smbh_df[idx:] += 1
                            
                                else:  # STAR - STAR
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
        
        x_fit = np.linspace(0, 100, 1000)
        # Each configuration separately
        for label in range(len(config_name)):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel(r"$t$ [kyr]", fontsize=self.AXLABEL_SIZE)
            ax.set_ylabel(r"$N_{\rm coll}$", fontsize=self.AXLABEL_SIZE)
            
            for i in range(len(data_array)-1):
                i += 1
                if i == 0:
                    ax.plot(time_array, data_array[i][label][1], 
                            color=self.cmap_colours[i],
                            alpha=0.3, lw=1)
                    ax.plot(time_array, data_array[i][label][2], 
                            color=self.cmap_colours[i],
                            alpha=0.3,lw=1)
                    ax.fill_between(
                        time_array, 
                        data_array[i][label][1],
                        data_array[i][label][2], 
                        color=self.cmap_colours[i], 
                        alpha=0.3
                    )
                    
                ax.plot(
                    time_array, data_array[i][label][0], 
                    color=self.cmap_colours[i],
                    label=data_labels[i], zorder=1, 
                    ls=linestyle[i]
                )
                
                    
            ax.legend(fontsize=self.TICK_SIZE)
            ax = self.tickers(ax, "plot", True)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
            ax.set_xlim(1.e-3, 100)
            ax.set_ylim(1.e-3, 1.05*data_array[3][label][1][-1])
            plt.savefig(f"plot/figures/Ncoll_vs_time_{config_name[label]}.pdf", dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(r"$t$ [kyr]", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$N_{\rm coll}$", fontsize=self.AXLABEL_SIZE)
        for label in range(len(config_name)):
            
            if label == 0 or label == 2:
                lw = 2
                ls = "-"
            else:
                lw = 1
                ls = "-."
                
            median_smoothed = moving_average(data_array[0][label][0], 20)
            IQRH_smoothed = moving_average(data_array[0][label][1], 20)
            IQRL_smoothed = moving_average(data_array[0][label][2], 20)
            time_smoothed = np.linspace(0, 100, len(IQRH_smoothed))
            
            if label == 0 or label == 1:
                MSMBH = 1e5 | units.MSun
            else:
                MSMBH = 4e5 | units.MSun
            if label == 0 or label == 2:
                VKICK = 300 | units.kms
            else:
                VKICK = 600 | units.kms
                
            params, covariance = curve_fit(custom_function, time_smoothed, median_smoothed, p0=[0], maxfev=10000)
            y_fit = custom_function(x_fit, *params)
            ax.plot(x_fit, y_fit, color=self.cmap_colours[i], ls="--", lw=1)
            print(f"Best fit: {params}")
            
            print((median_smoothed[-1]/100) * (1/np.log(MSMBH.value_in(units.MSun))) * (VKICK**3/MSMBH) * (MSMBH/VKICK**2)**2.5)
            
            
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
        ax = self.tickers(ax, "plot", True)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.set_xlim(1.e-3, 100)
        ax.set_ylim(1.e-3, data_array[0][2][1][-1])
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
            periapse_array = [ ]
            for id, run in enumerate(config):
                coll_sma_array = {
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
                            coll_sma_array[ID].append(np.log10(ke[2].value_in(units.pc)))
                            coll_periapse_array[ID].append(ke[7].value_in(units.deg)) 
                
                sma_array.append(coll_sma_array)
                periapse_array.append(coll_periapse_array)
            
            return [sma_array, periapse_array]
        
        def compute_dperi(periapse_array, sma_array, SMBH_mass):
            peri_array = [ ]
            for run_id, colliders in enumerate(periapse_array):
                for ID, peri_list in colliders.items():
                    if len(peri_list) < 2:
                        continue
                    
                    sma = sma_array[run_id][ID]
                    orb_period = np.sqrt((10**sma[0] | units.pc)**3 / (constants.G * SMBH_mass))
                    
                    # Loop through consecutive periapse values to calculate dperi
                    for p in range(len(peri_list) - 1):
                        delta_peri = abs(peri_list[p + 1] - peri_list[p])
                        if delta_peri > 0 | units.deg:
                            dperi = (dTime/orb_period) * (delta_peri)**-1
                            peri_array.append(dperi)
            return peri_array
        
        m4e5_300kms_colls = self.extract_folder("4e5", 300, "coll_orbital")
        m1e5_300kms_colls = self.extract_folder("1e5", 300, "coll_orbital")
        coll_id_4e5 = extract_peri_coll(m4e5_300kms_colls)
        coll_id_1e5 = extract_peri_coll(m1e5_300kms_colls)
        
        m4e5_300kms_snaps = self.extract_folder("4e5", 300, "simulation_snapshot")
        m1e5_300kms_snaps = self.extract_folder("1e5", 300, "simulation_snapshot")
        sma_4e5, periapse_4e5 = create_data_frames(m4e5_300kms_snaps, coll_id_4e5)
        sma_1e5, periapse_1e5 = create_data_frames(m1e5_300kms_snaps, coll_id_1e5)
        
        dTime = 100 | units.yr
        dperi_4e5 = compute_dperi(periapse_4e5, sma_4e5, SMBH_mass=4e5 | units.MSun)
        dperi_1e5 = compute_dperi(periapse_1e5, sma_1e5, SMBH_mass=1e5 | units.MSun)
        print(len(dperi_1e5), len(dperi_4e5))
        
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$( P_{\rm orb}\ \log_{10}\langle|\dot{\omega}|\rangle)^{-1}$ [$1/^{\circ}$]", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$f_{<}$", fontsize=self.AXLABEL_SIZE)
        for i, dperi in enumerate([dperi_4e5, dperi_1e5]):
            x_sorted = np.sort(dperi)
            y_lens = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            ax.plot(np.log10(x_sorted), y_lens, lw=2, color=self.colours_two[i])
            ax.axhline(y_lens[abs(x_sorted - 1).argmin()], color=self.colours_two[i], ls="-.")
            ax.axhline(y_lens[abs(x_sorted - 10).argmin()], color=self.colours_two[i], ls=":")
        
        ax.scatter(None, None, color="red", label=r"$10^{5}$ M$_\odot$")
        ax.scatter(None, None, color="blue", label=r"$4\times10^{5}$ M$_\odot$")
        ax.set_xlim(-1.3, 3.5)
        ax.set_ylim(0, 1.01)
        ax.legend(fontsize=self.TICK_SIZE)
        self.tickers(ax, "plot", False)
        fname = f"plot/figures/CDF_dperi.pdf"
        plt.savefig(fname, dpi=250, bbox_inches='tight')
        plt.close()
        plt.clf()
        
plot = NCSCPlotter()
plot.plot_GW_vs_time()
STOP
plot.dperi_colls()
plot.sma_ecc_traj_colls()