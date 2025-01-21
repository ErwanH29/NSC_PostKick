import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as mticker
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import matplotlib.patheffects as path_effects
import natsort
import numpy as np
from scipy.stats import gaussian_kde
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
        
        black_rgba = matplotlib.colors.to_rgba("black")
        cmap = matplotlib.colormaps['cool']
        self.cmap_colours = np.vstack((black_rgba, cmap(np.linspace(0.1, 1, 3))))
        self.ls = ["-", "-."]
        
    def extract_folder(self, SMBH_mass, vkick, folder):
        """Extract the data folders"""
        data_folders = natsort.natsorted(glob.glob(f"/media/erwanh/Elements/BH_Post_Kick/data/{vkick}kms_m{SMBH_mass}/Nimbh0_RA_BH_Run/{folder}/*"))
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
        ecc_range = np.linspace(0, 1, 10000)
        
        tGW = 10 | units.kyr
        avg_star_mass = 3.8 | units.MSun
        mu = SMBH_mass * avg_star_mass
        coefficient = 256*constants.G**3/(5*constants.c**5)*(mu*(SMBH_mass + avg_star_mass))*tGW
        sma_range = [((coefficient/(1-i**2)**(7./2.))**(1/4)).value_in(units.pc) for i in ecc_range]
        
        return ecc_range, sma_range
    
    def ecc_sma_tidal(self, mass, SMBH_mass):
        ecc_range = np.linspace(0, 1, 10000)
        radius = self.ZAMS_radius(mass)
        rtide = radius * (0.844**2 * SMBH_mass/(mass))**(1./3.)
        sma_range = [(rtide/(1-i)).value_in(units.pc) for i in ecc_range]
        
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
            sys.stdout.write(f"\rProgress: {str(100*i/len(minor))[:5]}%")
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
        rinfl = 22 * (mSMBH/(1e8 | units.MSun))**0.55  | units.pc #  Equation 14, for cusp galaxies (MSMBH < 1e8 MSun)
        rinfl = 0.23 | units.pc
        print(rinfl.value_in(units.pc))
        nTDE = 1.5e-6 * (mSMBH/(1e7 | units.MSun)) * (rinfl/(10 | units.pc))**-2 * (vkick/(1000 | units.kms))**-1 | units.yr**-1
        total_TDE = nTDE * time
        return total_TDE
    
    def plot_GW_vs_time(self):
        """Plot GW events occuring in time"""
        
        BIN_RESOLUTION = 2000
        TIME_PER_BIN = 10 | units.yr
        time_array = np.linspace(10**-4, 20, BIN_RESOLUTION)
        
        m1e5_300kms = self.extract_folder("1e5", 300, "coll_orbital")
        m1e5_600kms = self.extract_folder("1e5", 600, "coll_orbital")
        m4e5_300kms = self.extract_folder("4e5", 300, "coll_orbital")
        m4e5_600kms = self.extract_folder("4e5", 600, "coll_orbital")
        
        data_configs = [m1e5_300kms, m1e5_600kms, m4e5_300kms, m4e5_600kms]
        config_name = ["1e5_300kms", "1e5_600kms", "4e5_300kms", "4e5_600kms"]
        vkick_arr = [300, 600] | units.kms
        msmbh_arr = [1e5, 4e5] | units.MSun
        
        coll_events_arr = [ ]
        emri_events_arr = [ ]
        tde_events_arr = [ ]
        tde_smbh_events_arr = [ ]
        ss_events_arr = [ ]
        for i, IC_params in enumerate(data_configs):
            coll_events_run = [ ]
            emri_events_run = [ ]
            tde_events_run = [ ]
            tde_smbh_events_run = [ ]
            ss_events_run = [ ]
            
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
                            
                            print(type_a, type_b)
                            
                            idx = int(tcoll/TIME_PER_BIN)
                            coll_events_df[idx:] += 1
                            if type_a == 14: # Type a is BH
                                if type_b == 14:
                                    if max(mass_a/mass_b, mass_b/mass_a) > 1e4:
                                        emri_events_df[idx:] += 1
                                    else:
                                        GW_events_df[idx:] += 1
                                elif type_b >= 10:  # Type b is compact object
                                    GW_events_df[idx:] += 1
                                else:
                                    tde_events_df[idx:] += 1
                            
                            elif type_a >= 10:  # Type a is compact object
                                if type_b == 14:
                                    if max(mass_a/mass_b, mass_b/mass_a) > 1e4:
                                        emri_events_df[idx:] += 1
                                    else:
                                        GW_events_df[idx:] += 1
                                elif type_b >= 10:
                                    GW_events_df[idx:] += 1
                                else:
                                    tde_events_df[idx:] += 1
                            
                            # Type a = Star
                            elif type_b >= 10:  
                                tde_events_df[idx:] += 1
                            
                            else:
                                ss_events_df[idx:] += 1
                                
                    coll_events_run.append(coll_events_df)
                    emri_events_run.append(emri_events_df)
                    tde_events_run.append(tde_events_df)
                    tde_smbh_events_run.append(tde_smbh_df)
                    ss_events_run.append(ss_events_df)
                    
            if len(coll_events_run) > 0:
                mean_coll = np.mean(coll_events_run, axis=0)
                IQRH_coll = np.percentile(coll_events_run, 75, axis=0)
                IQL_coll = np.percentile(coll_events_run, 25, axis=0)
                
                mean_emri = np.mean(emri_events_run, axis=0)
                IQRH_emri = np.percentile(emri_events_run, 75, axis=0)
                IQL_emri = np.percentile(emri_events_run, 25, axis=0)
                
                mean_tde = np.mean(tde_events_run, axis=0)
                IQRH_tde = np.percentile(tde_events_run, 75, axis=0)
                IQL_tde = np.percentile(tde_events_run, 25, axis=0)
                
                mean_tde_smbh = np.mean(tde_smbh_events_run, axis=0)
                IQRH_tde_smbh = np.percentile(tde_smbh_events_run, 75, axis=0)
                IQRL_tde_smbh = np.percentile(tde_smbh_events_run, 25, axis=0)
                
                mean_ss = np.mean(ss_events_run, axis=0)
                IQRH_ss = np.percentile(ss_events_run, 75, axis=0)
                IQL_ss = np.percentile(ss_events_run, 25, axis=0)
                
            else:
                mean_coll = IQRH_coll = IQRL_coll = np.zeros(BIN_RESOLUTION)
                mean_emri = IQRH_emri = IQL_emri = np.zeros(BIN_RESOLUTION)
                mean_tde = IQRH_tde = IQL_tde = np.zeros(BIN_RESOLUTION)
                mean_tde_smbh = IQRH_tde_smbh = IQRL_tde_smbh = np.zeros(BIN_RESOLUTION)
                mean_ss = IQRH_ss = IQL_ss = np.zeros(BIN_RESOLUTION)
            
            coll_events_arr.append([mean_coll, IQRH_coll, IQL_coll])
            emri_events_arr.append([mean_emri, IQRH_emri, IQL_emri])
            tde_events_arr.append([mean_tde, IQRH_tde, IQL_tde])
            tde_smbh_events_arr.append([mean_tde_smbh, IQRH_tde_smbh, IQRL_tde_smbh])
            ss_events_arr.append([mean_ss, IQRH_ss, IQL_ss])
            
        data_labels = [r"$N_{\rm coll}$", r"$N_{\rm EMRI}$", r"$N_{\rm TDE}$", r"$N_{\rm **}$"]
        data_array = [coll_events_arr, emri_events_arr, tde_events_arr, ss_events_arr, tde_smbh_events_arr]
        
        # Each configuration separately
        for label in range(len(config_name)):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel(r"$t$ [kyr]", fontsize=self.AXLABEL_SIZE)
            ax.set_ylabel(r"$N_{\rm coll}$", fontsize=self.AXLABEL_SIZE)
            
            vkick = vkick_arr[label%2]
            msmbh = msmbh_arr[label//2]
            NTDE = [self.stone_TDE_rate(msmbh, vkick, 10. | units.yr) for time in range(BIN_RESOLUTION)]
            NTDE_array = [ ]
            total_TDE = 0
            for N in NTDE:
                N /= 0.14
                NTDE_array.append(N + total_TDE)
                total_TDE += N
            
            print(f"{config_name[label]}")
            for i in range(len(data_array)):
                if i == 0:
                    print(f"{data_labels[i]}= {data_array[i][label][0][-1]},")
                    print(f"+{data_array[i][label][2][-1]}, -{data_array[i][label][1][-1]}")
                    ax.plot(time_array, data_array[i][label][0], 
                            color=self.cmap_colours[i],
                            label=data_labels[i],
                            lw=1, zorder=2, ls="-.")
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
                    alpha=0.3)
                elif i != (len(data_array) - 1):
                    print(f"{data_labels[i]}= {data_array[i][label][0][-1]}")
                    ax.plot(time_array, data_array[i][label][0], 
                            color=self.cmap_colours[i],
                            label=data_labels[i],
                            lw=3, zorder=1)
                else:
                    ax.plot(time_array, data_array[i][label][0], 
                            color=self.cmap_colours[2],
                            lw=2, ls=":", zorder=1)
                    
            ax.legend(fontsize=self.TICK_SIZE)
            ax = self.tickers(ax, "plot", True)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
            ax.set_xlim(1.e-3, 20)
            ax.set_ylim(1.e-3, data_array[0][label][1][-1])
            plt.savefig(f"plot/figures/Ncoll_vs_time_{config_name[label]}.pdf", dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(r"$t$ [kyr]", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$N_{\rm coll}$", fontsize=self.AXLABEL_SIZE)
        for label in range(len(config_name)):
            ax.plot(time_array, data_array[0][label][0], 
                    color=self.colours[label//2],
                    ls=self.ls[label%2],
                    lw=3)
            if label == 0 or label == 2:
                IQRH_smoothed = moving_average(data_array[i][label][1], 30)
                IQRL_smoothed = moving_average(data_array[i][label][2], 30)
                time_smoothed = np.linspace(0, 20, len(IQRH_smoothed))
                
                ax.plot(time_smoothed, IQRH_smoothed, 
                        color=self.colours[label//2],
                        alpha=0.3, lw=1)
                ax.plot(time_smoothed, IQRL_smoothed, 
                        color=self.colours[label//2],
                        alpha=0.3,lw=1)
                ax.fill_between(
                    time_smoothed, 
                    IQRL_smoothed,
                    IQRH_smoothed, 
                    color=self.colours[label//2],
                    alpha=0.3
                )
        ax.scatter(None, None, color="red", label=r"$M_{\rm SMBH} = 10^{5}$ M$_\odot$")
        ax.scatter(None, None, color="blue", label=r"$M_{\rm SMBH} = 4\times10^{5}$ M$_\odot$")
                    
        ax.legend(fontsize=self.TICK_SIZE)
        ax = self.tickers(ax, "plot", True)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.set_xlim(1.e-3, 20)
        ax.set_ylim(1.e-3, 74)#data_array[0][2][1][-1])
        plt.savefig(f"plot/figures/Ncoll_vs_time_all.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
        
    def plot_cluster_evolution(self):
        """Plot the half-mass radius and population evolution of the NSC and HCSC"""
        
        m1e5_300kms = self.extract_folder("1e5", 300, "simulation_snapshot")
        m1e5_600kms = self.extract_folder("1e5", 600, "simulation_snapshot")
        m4e5_300kms = self.extract_folder("4e5", 300, "simulation_snapshot")
        m4e5_600kms = self.extract_folder("4e5", 600, "simulation_snapshot")
        
        data_configs = [m1e5_300kms, m1e5_600kms, m4e5_300kms, m4e5_600kms]
        config_name = ["1e5_300kms", "1e5_600kms", "4e5_300kms", "4e5_600kms"]
        levels = [1e-2, 1e-1, 0.5, 0.9]
        
        for i, IC_params in enumerate(data_configs):
            print(f"Configuration: {config_name[i]}")
            
            sma_df = [ ]
            ecc_df = [ ]
            inc_df = [ ]
            for iter, run in enumerate(IC_params):
                data_files = natsort.natsorted(glob.glob(f"{run}/*"))
                if len(data_files) > 900:
                    print(f"...Contour for {data_files[-1]}...")
                    particle_set = read_set_from_file(data_files[-1], format="amuse")
                    SMBH = particle_set[particle_set.mass.argmax()]
                    minor = particle_set - SMBH

                    lag = LagrangianRadii(particle_set)[-2]
                    for j, p in enumerate(minor):
                        sys.stdout.write(f"\rProgress: {str(100*j/len(minor))[:5]}%")
                        sys.stdout.flush()
                        
                        rij = p.position - SMBH.position
                        vij = p.velocity - SMBH.velocity
                        trajectory = (np.dot(rij, vij))/(rij.length() * vij.length()) 
                        
                        bin_sys = Particles()
                        bin_sys.add_particle(SMBH)
                        bin_sys.add_particle(p)
                        
                        ke = orbital_elements(bin_sys, G=constants.G)
                        sma = ke[2]
                        ecc = ke[3]
                        true_anom = ke[4]
                        inc = ke[5]
                        arg_periapsis = ke[7]
                        if trajectory > 0 and rij.length() > 1.5*lag and ecc > 1:
                            continue
                        
                        sma_df.append(np.log10(sma.value_in(units.pc)))
                        ecc_df.append(ecc)
                        inc_df.append(inc.value_in(units.deg))
        
            values = np.vstack([sma_df, ecc_df])
            xx, yy = np.mgrid[-4:0:400j, 0:1:400j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            kernel = gaussian_kde(values, bw_method=0.2)
            f = np.reshape(kernel(positions).T, xx.shape)
            f_min, f_max = np.min(f), np.max(f)
            fnorm = (f - f_min) / (f_max - f_min)
                
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel(r"$\log_{10}a$ [pc]", fontsize=self.AXLABEL_SIZE)
            ax.set_ylabel(r"$e$", fontsize=self.AXLABEL_SIZE)
            self.tickers(ax, "plot", False)
            ax.contourf(xx, yy, fnorm, cmap="Blues", levels=levels, zorder=1, extend="max")
            cset = ax.contour(xx, yy, fnorm, colors="k", levels=levels, zorder=2)
            
            ax.clabel(cset, inline=1, fontsize=10)
            ax.set_xlim(-4, 0)
            ax.set_ylim(0, 1)
            plt.savefig(f"plot/figures/final_contour_sma_ecc_{config_name[i]}.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            
            values = np.vstack([sma_df, inc_df])
            xx, yy = np.mgrid[-4:0:400j, -180:180:400j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            kernel = gaussian_kde(values, bw_method=0.2)
            f = np.reshape(kernel(positions).T, xx.shape)
            f_min, f_max = np.min(f), np.max(f)
            fnorm = (f - f_min) / (f_max - f_min)
                
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel(r"$\log_{10}a$ [pc]", fontsize=self.AXLABEL_SIZE)
            ax.set_ylabel(r"$i$ [$^{\circ}$]", fontsize=self.AXLABEL_SIZE)
            self.tickers(ax, "plot", False)
            ax.contourf(xx, yy, fnorm, cmap="Blues", levels=levels, zorder=1, extend="max")
            cset = ax.contour(xx, yy, fnorm, colors="k", levels=levels, zorder=2)
            
            ax.clabel(cset, inline=1, fontsize=10)
            ax.set_xlim(-4, 0)
            ax.set_ylim(-180, 180)
            plt.savefig(f"plot/figures/final_contour_sma_inc_{config_name[i]}.pdf", dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_cluster_initial(self):
        """Plot the system in x, y coordinates"""
        initial_bodies = read_set_from_file("/media/erwanh/Elements/BH_Post_Kick/600kms_m1e5/Nimbh0_RA_BH_Run/init_snapshot/config_1_bound.hdf5", "hdf5")
        initial_bodies.position -= initial_bodies[initial_bodies.mass.argmax()].position
        
        levels = [1e-2, 1e-1, 0.5]
        values = np.vstack([initial_bodies.x.value_in(units.pc), 
                            initial_bodies.y.value_in(units.pc)])
        xx, yy = np.mgrid[-0.044:0.044:500j, -0.044:0.044:500j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kernel = gaussian_kde(values, bw_method="silverman")#, bw_method=0.)
        f = np.reshape(kernel(positions).T, xx.shape)
        f_min, f_max = np.min(f), np.max(f)
        fnorm = (f - f_min) / (f_max - f_min)
        

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        self.tickers(ax, "plot", False)
        ax.set_xlim(-0.044, 0.044)
        ax.set_ylim(-0.044, 0.044)
        ax.set_xlabel(r"$x$ [pc]", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$y$ [pc]", fontsize=self.AXLABEL_SIZE)

        cset = ax.contour(xx, yy, fnorm, colors="k", levels=levels, zorder=1)
        ax.scatter(initial_bodies[initial_bodies.mass.argmax()].x.value_in(units.pc), 
                    initial_bodies[initial_bodies.mass.argmax()].y.value_in(units.pc), 
                    color="black", s=5, zorder=3)

        fname = "plot/figures/HCSC_system_plot.png"
        plt.savefig(fname, dpi=250, bbox_inches='tight')
        plt.close()
        plt.clf()
    
    def sma_ecc_traj_colls(self):
        """Plot collision trajectory of a specific run"""
        
        run_idx = 2
        
        m1e5_300kms_colls = self.extract_folder("1e5", 300, "coll_orbital")[run_idx]
        m4e5_600kms_colls = self.extract_folder("4e5", 600, "coll_orbital")[run_idx]
        
        coll_data_configs = [m1e5_300kms_colls, m4e5_600kms_colls]
        colliders_id = [ ]
        final_ecc = [ ]
        final_sma = [ ]
        for i, IC_params in enumerate(coll_data_configs):
            data_files = natsort.natsorted(glob.glob(f"{IC_params}/*"))
            colliders_id_df = [ ]
            final_ecc_df = [ ]
            final_sma_df = [ ]
            for file in data_files:
                with open(file, 'rb') as df:
                    lines = [x.decode('utf8').strip() for x in df.readlines()]
                    key_a = int(lines[1].split("[")[-1].split("]")[0])
                    key_b = int(lines[2].split("[")[-1].split("]")[0])
                    mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                    mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun
                    
                    sma = float(lines[6].split(": ")[1][:-3]) | units.au
                    ecc = float(lines[7].split(": ")[1])
                    final_sma_df.append(np.log10(sma.value_in(units.pc)))
                    final_ecc_df.append(np.log10(1-ecc))
                    
                    if mass_a > mass_b:
                        colliders_id_df.append(key_b)
                    else:
                        colliders_id_df.append(key_a)
            colliders_id.append(colliders_id_df)
            final_ecc.append(final_ecc_df)
            final_sma.append(final_sma_df)
        
        m1e5_300kms_snaps = self.extract_folder("1e5", 300, "simulation_snapshot")
        m4e5_300kms_snaps = self.extract_folder("4e5", 300, "simulation_snapshot")
        snap_data_configs = [m1e5_300kms_snaps, m4e5_300kms_snaps] 
        
        all_sma_init = [ ]
        all_ecc_init = [ ]
        colls_sma = [ ]
        colls_ecc = [ ]
        for i, IC_params in enumerate(snap_data_configs):
            sma_df = [ ]
            ecc_df = [ ]
            coll_sma_df = [[ ] for i in range(len(colliders_id[i]))]
            coll_ecc_df = [[ ] for i in range(len(colliders_id[i]))]
            for j, run in enumerate(IC_params):
                print(run)
                data_files = natsort.natsorted(glob.glob(f"{run}/*"))
                if len(data_files) > 0:
                    if j != run_idx:
                        initial_set = read_set_from_file(data_files[0], format="amuse")
                        SMBH = initial_set[initial_set.mass.argmax()]
                        minor = initial_set - SMBH
                        
                        for p in minor:
                            bin_system = Particles()
                            bin_system.add_particle(SMBH)
                            bin_system.add_particle(p)
                            ke = orbital_elements(bin_system, G=constants.G)
                            sma_df.append(np.log10(ke[2].value_in(units.pc)))
                            ecc_df.append(ke[3])
                            
                    else:
                        for k, dt in enumerate(data_files):
                            particles = read_set_from_file(dt, format="amuse")
                            SMBH = particles[particles.mass.argmax()]
                            
                            if k == 0:
                                minor = particles - SMBH
                                for p in minor:
                                    bin_system = Particles()
                                    bin_system.add_particle(SMBH)
                                    bin_system.add_particle(p)
                                    
                                    ke = orbital_elements(bin_system, G=constants.G)
                                    sma_df.append(np.log10(ke[2].value_in(units.pc)))
                                    ecc_df.append(np.log10(1-ke[3]))
                                    
                            for l, ID in enumerate(colliders_id[i]):
                                target_minor = particles[particles.key == ID]
                                if (target_minor):
                                    bin_system = Particles()
                                    bin_system.add_particle(SMBH)
                                    bin_system.add_particle(target_minor)
                                    
                                    ke = orbital_elements(bin_system, G=constants.G)
                                    coll_sma_df[l].append(np.log10(ke[2].value_in(units.pc)))
                                    coll_ecc_df[l].append(np.log10(1-ke[3]))              
            all_sma_init.append(sma_df)
            all_ecc_init.append(ecc_df)            
            colls_sma.append(coll_sma_df)
            colls_ecc.append(coll_ecc_df)
            
        time = np.linspace(0, 10, 1001)
        norm = mcolors.Normalize(vmin=time.min(), vmax=time.max())
        cmap = cm.viridis  # Choose a colormap
        
        levels = [1e-2, 1e-1, 0.5, 0.9]
        min_sma = min(min(all_sma_init[0]), min(all_sma_init[1]))
        min_ecc = min(min(all_ecc_init[0]), min(all_ecc_init[1]))
        
        labs = ["m1e5_300", "m4e5_300"]
        smbh_masses = [1e5, 4e5] | units.MSun
        for data in range(2):
            values = np.vstack([all_sma_init[data], all_ecc_init[data]])
            xx, yy = np.mgrid[min_sma:1:500j, min_ecc:0:500j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            kernel = gaussian_kde(values, bw_method = "silverman")
            f = np.reshape(kernel(positions).T, xx.shape)
            f_min, f_max = np.min(f), np.max(f)
            fnorm = (f - f_min) / (f_max - f_min)
            
            ecc_range_GW, sma_range_GW = self.ecc_sma_GW(smbh_masses[data])
            ecc_range_tidal, sma_range_tidal = self.ecc_sma_tidal(1 | units.Msun, smbh_masses[data])
            ecc_range_tidal, sma_range_tidal_10 = self.ecc_sma_tidal(10 | units.Msun, smbh_masses[data])

            fig, ax = plt.subplots(figsize=(6, 6))
            self.tickers(ax, "plot", False)
            ax.set_xlim(min_sma-0.3, 1)
            ax.set_ylim(-4, 0)
            ax.set_xlabel(r"$\log_{10}a$ [pc]", fontsize=self.AXLABEL_SIZE)
            ax.set_ylabel(r"$\log_{10}(1-e)$", fontsize=self.AXLABEL_SIZE)
            ax.contourf(xx, yy, fnorm, cmap="Blues", levels=levels, zorder=1, extend="max")
            cset = ax.contour(xx, yy, fnorm, colors="k", levels=levels, zorder=2)
            for sma, ecc in zip(colls_sma[data], colls_ecc[data]):
                for i in range(len(sma)):
                    ax.scatter(sma[i], ecc[i], color=cmap(norm(time[i])), s=1, zorder=3)
            ax.scatter(final_sma[data], final_ecc[data], color="red", 
                        edgecolors="black", marker="X", 
                        zorder=4)
            
            ax.plot(np.log10(sma_range_GW), np.log10(1-ecc_range_GW), color="red", ls=":", zorder=5)
            ax.plot(np.log10(sma_range_tidal), np.log10(1-ecc_range_tidal), color="black", ls=":", lw=2, zorder=5)
            ax.plot(np.log10(sma_range_tidal_10), np.log10(1-ecc_range_tidal), color="black", ls="-.", lw=2, zorder=5)
            
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Required for ScalarMappable
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(r'$t$ [kyr]', fontsize=self.AXLABEL_SIZE)
            
            fname = f"plot/figures/coll_traj_{labs[data]}.png"
            plt.savefig(fname, dpi=250, bbox_inches='tight')
            plt.close()
            plt.clf()
                        
        
plot = NCSCPlotter()
plot.plot_GW_vs_time()
#plot.plot_cluster_initial()
#plot.plot_cluster_evolution()
#plot.sma_ecc_traj_colls()