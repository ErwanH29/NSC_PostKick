import glob
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as mtick
import natsort
import numpy as np

from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import constants, units, Particles, read_set_from_file


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


class HyperbolicPlotter(object):
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
        
    def tickers(self, ax, ptype):
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
        
    def extract_folder(self, SMBH_mass, vkick):
        """Extract the data folders"""
        folder_prefix = f"/media/erwanh/PhD Material/All_Data/3_Runaway_BH_At_Kick/{vkick}kms_m{SMBH_mass}"
        merge_txt = natsort.natsorted(glob.glob(f"{folder_prefix}/Nimbh0_RA_BH_Run/coll_orbital/*"))
        merge_hdf5 = natsort.natsorted(glob.glob(f"{folder_prefix}/Nimbh0_RA_BH_Run/merge_snapshots/*"))
        return merge_txt, merge_hdf5
    
    def make_planar(self, sma, ecc, true_anomaly, Omega, omega, inc):
        """
        Function to make a planar orbit
        Args:
            sma (float): Semi-major axis
            ecc (float): Eccentricity
            true_anomaly (float): True anomaly
        Returns:
            x, y (float):  x and y coordinates of the orbit
        """
        r = (sma * (1 - ecc**2)) / (1 + ecc*np.cos(true_anomaly))
        x = r * np.cos(true_anomaly)
        y = r * np.sin(true_anomaly)
        
        cos_loa = np.cos(Omega)
        sin_loa = np.sin(Omega)
        cos_aop = np.cos(omega)
        sin_aop = np.sin(omega)
        cos_inc = np.cos(inc)
        sin_inc = np.sin(inc)
        
        x = (cos_loa * cos_aop - sin_loa * sin_aop * cos_inc) * x \
            + (-cos_loa * sin_aop - sin_loa * cos_aop * cos_inc) * y
        y = (sin_loa * cos_aop + cos_loa * sin_aop * cos_inc) * x \
            + (-sin_loa * sin_aop + cos_loa * cos_aop * cos_inc) * y
        z = (sin_aop * sin_inc) * x + (cos_aop * sin_inc) * y
        
        return x, y, z
    
    def sma_ecc_hyperbolic_colls(self):
        vkicks = [300, 600]
        SMBH_masses = ["1e5", "4e5"]
        
        for kick in vkicks:
            for SMBH_Mass in SMBH_masses:
                configs_txt, configs_hdf5 = self.extract_folder(SMBH_Mass, kick)
                
                merger_sma_df = [ ]
                merger_ecc_df = [ ]
                merger_dr_df = [ ]
                merger_x_df = [ ]
                merger_y_df = [ ]
                merger_z_df = [ ]
                
                smbh_sma_df = [ ]
                smbh_ecc_df = [ ]
                smbh_tan_df = [ ]
                smbh_inc_df = [ ]
                smbh_loa_df = [ ]
                smbh_aop_df = [ ]
                for config_txt, config_hdf5 in zip(configs_txt, configs_hdf5):
                    text_file = natsort.natsorted(glob.glob(f"{config_txt}/*"))
                    hdf5_file = natsort.natsorted(glob.glob(f"{config_hdf5}/*"))
                    for txt, hdf5 in zip(text_file, hdf5_file):
                        with open(txt, 'rb') as df:
                            lines = [x.decode('utf8').strip() for x in df.readlines()]
                            key1 = lines[1].split("[")[-1][:-1]
                            key2 = lines[2].split("[")[-1][:-1]
                            mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                            mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun
                            type_a = int(lines[5].split("<")[1].split("- ")[0])
                            type_b = int(lines[5].split("<")[2].split("- ")[0])
                        
                        if max(mass_a, mass_b) > 1000 | units.MSun:
                            continue
                        if min(type_a, type_b) < 10:
                            continue
                        
                        particle_set = read_set_from_file(hdf5, "amuse")
                        SMBH = particle_set[particle_set.mass.argmax()]
                        particle_set.position -= SMBH.position
                        particle_set.velocity -= SMBH.velocity
                        
                        coll_a = particle_set[particle_set.key == int(key1)]
                        coll_b = particle_set[particle_set.key == int(key2)]
                        dr = (coll_a.position - coll_b.position).lengths()
                        radius = (coll_a.radius + coll_b.radius)
                        if (0):
                            print(f"Type_a {type_a}, Type_b {type_b}, ")
                            print(f"Mass_a {mass_a.in_(units.MSun)}, Mass_b {mass_b.in_(units.MSun)} ")
                            print(f"dr/radius = {dr/radius}")
                        
                        bin = coll_a + coll_b
                        ke = orbital_elements(bin, G=constants.G)
                        merger_sma_df.append(np.log10(abs(ke[2]).value_in(units.au)))
                        merger_ecc_df.append(np.log10(ke[3]))
                        
                        for coll_ in [coll_a, coll_b]:
                            bin = SMBH + coll_
                            ke = orbital_elements(bin, G=constants.G)
                            smbh_sma_df.append(np.log10(ke[2].value_in(units.au)))
                            smbh_ecc_df.append(ke[3])
                            smbh_tan_df.append(ke[4].value_in(units.rad))
                            smbh_inc_df.append(ke[5].value_in(units.rad))
                            smbh_loa_df.append(ke[6].value_in(units.rad))
                            smbh_aop_df.append(ke[7].value_in(units.rad))
                            merger_dr_df.append(dr.value_in(units.au))
                            merger_x_df.append(coll_.x.value_in(units.au))
                            merger_y_df.append(coll_.y.value_in(units.au))
                            merger_z_df.append(coll_.z.value_in(units.au))
                            
                        bin = coll_a + coll_b
                        com_particle = Particles(1)
                        com_particle.mass = bin.mass.sum()
                        com_particle.velocity = bin.center_of_mass_velocity()
                        com_particle.position = bin.center_of_mass()
                        
                        bin = Particles(particles=[com_particle, SMBH])
                        ke = orbital_elements(bin, G=constants.G)
                        smbh_sma_df.append(np.log10(ke[2].value_in(units.au)))
                        smbh_ecc_df.append(ke[3])
                        smbh_tan_df.append(ke[4].value_in(units.rad))
                        smbh_inc_df.append(ke[5].value_in(units.rad))
                        smbh_loa_df.append(ke[6].value_in(units.rad))
                        smbh_aop_df.append(ke[7].value_in(units.rad))
                        merger_dr_df.append(dr.value_in(units.au))
                        
                fig, ax = plt.subplots()
                ax.set_xlabel(r"$\log_{10} a_{\rm coll}$ [AU]", fontsize=self.AXLABEL_SIZE)
                ax.set_ylabel(r"$\log_{10} e_{\rm coll}$", fontsize=self.AXLABEL_SIZE)
                ax.scatter(merger_sma_df, merger_ecc_df, c="black", s=2, alpha=0.5)
                self.tickers(ax, "plot")
                plt.savefig(f"plot/figures/{kick}kms_{SMBH_Mass}_SBH_Mergers_sma_ecc.pdf", dpi=300, bbox_inches="tight")
                plt.clf()
                plt.close()
                
                fig, ax = plt.subplots()
                ax.set_xlabel(r"$\log_{10} a_{\rm IMBH}$ [AU]", fontsize=self.AXLABEL_SIZE)
                ax.set_ylabel(r"$\log_{10} e_{\rm IMBH}$", fontsize=self.AXLABEL_SIZE)
                ax.scatter(smbh_sma_df, smbh_ecc_df, c="black", s=2, alpha=0.5)
                for i in range(0, len(smbh_ecc_df), 3):
                    if i+1 < len(smbh_ecc_df):
                        ax.plot(
                            [smbh_sma_df[i], smbh_sma_df[i+1]], 
                            [smbh_ecc_df[i], smbh_ecc_df[i+1]], 
                            c="black", lw=0.5, alpha=0.5
                        )
                self.tickers(ax, "plot")
                plt.savefig(f"plot/figures/{kick}kms_{SMBH_Mass}_Param_w_IMBH_sma_ecc.pdf", dpi=300, bbox_inches="tight")
                plt.clf()
                plt.close()
                
                fig, ax = plt.subplots()
                ax.set_xlabel(r"$\log_{10} dr_{ij}$ [au]", fontsize=self.AXLABEL_SIZE)
                ax.set_ylabel(r"$\nu$ [$^{\circ}$]", fontsize=self.AXLABEL_SIZE)
                ax.scatter(merger_dr_df, np.array(smbh_tan_df) * 180/np.pi, c="black", s=2, alpha=0.5)
                self.tickers(ax, "plot")
                plt.savefig(f"plot/figures/{kick}kms_{SMBH_Mass}_Param_w_IMBH_dr_true_anomaly.pdf", dpi=300, bbox_inches="tight")
                plt.clf()
                plt.close()
                
                cmap = plt.cm.get_cmap('tab20', int(len(smbh_ecc_df) / 2))
                theta_range = np.linspace(0, 2*np.pi, 5000)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                fig, ax = plt.subplots()
                for i in range(0, len(smbh_ecc_df), 3):
                    if i+1 < len(smbh_ecc_df):
                        colour = cmap(i//2)
                        
                        # Collider_a
                        sma_a = 10**smbh_sma_df[i]
                        ecc_a = smbh_ecc_df[i]
                        coll_a = smbh_tan_df[i]
                        aop_a = smbh_aop_df[i]
                        loa_a = smbh_loa_df[i]
                        inc_a = smbh_inc_df[i]
                        
                        x_a, y_a, z_a = self.make_planar(sma_a, ecc_a, theta_range, loa_a, aop_a, inc_a)
                        x_coll_a, y_coll_a, z_coll_a = self.make_planar(sma_a, ecc_a, coll_a, loa_a, aop_a, inc_a)
                        x_coll_a, y_coll_a, z_coll_a = merger_x_df[i], merger_y_df[i], merger_z_df[i]

                        # Collider_b
                        sma_b = 10**smbh_sma_df[i+1]
                        ecc_b = smbh_ecc_df[i+1]
                        coll_b = smbh_tan_df[i+1]
                        aop_b = smbh_aop_df[i+1]
                        loa_b = smbh_loa_df[i+1]
                        inc_b = smbh_inc_df[i+1]
                        
                        x_b, y_b, z_b = self.make_planar(sma_b, ecc_b, theta_range, loa_b, aop_b, inc_b)
                        x_coll_b, y_coll_b, z_coll_b = self.make_planar(sma_b, ecc_b, coll_b, loa_b, aop_b, inc_b)
                        x_coll_b, y_coll_b, z_coll_b = merger_x_df[i+1], merger_y_df[i+1], merger_z_df[i+1]

                        # Collider_com
                        sma_com = 10**smbh_sma_df[i+2]
                        ecc_com = smbh_ecc_df[i+2]
                        aop_com = smbh_aop_df[i+2]
                        loa_com = smbh_loa_df[i+2]
                        inc_com = smbh_inc_df[i+2]
                        
                        x_c, y_c, z_c = self.make_planar(sma_com, ecc_com, theta_range, loa_com, aop_com, inc_com)
                        
                        ax.plot(x_a, y_a, c=colour)
                        ax.plot(x_b, y_b, c=colour)
                        ax.plot(x_c, y_c, c=colour, ls=":")
                        ax.scatter(x_coll_a, y_coll_a, color=colour, marker="X")
                        ax.scatter(x_coll_b, y_coll_b, color=colour)
                        ax.scatter(0, 0, color="black", marker="o", s=100, label="SMBH")
                        
                plt.savefig(f"plot/figures/{kick}kms_{SMBH_Mass}_Coll_Locs.pdf", dpi=300, bbox_inches="tight")
                plt.clf()
                plt.close()
                
plot = HyperbolicPlotter()
plot.sma_ecc_hyperbolic_colls()