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
        folder_prefix = f"/media/erwanh/PhD Material/All_Data/3_Runaway_BH_At_Kick_Old/{vkick}kms_m{SMBH_mass}"
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
        true_anomalies = np.linspace(-np.pi, np.pi, 10000)
        x_list, y_list, z_list = [], [], []

        for nu in true_anomalies:
            r = sma * (1 - ecc**2) / (1 + ecc * np.cos(nu))  # Radius

            # Position in orbital plane
            x_orb = r * np.cos(nu)
            y_orb = r * np.sin(nu)
            z_orb = 0.0

            # Rotate to 3D space
            cos_Omega = np.cos(Omega)
            sin_Omega = np.sin(Omega)
            cos_omega = np.cos(omega)
            sin_omega = np.sin(omega)
            cos_inc = np.cos(inc)
            sin_inc = np.sin(inc)

            X = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_inc) * x_orb + \
                (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_inc) * y_orb
            Y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_inc) * x_orb + \
                (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_inc) * y_orb
            Z = (sin_omega * sin_inc) * x_orb + (cos_omega * sin_inc) * y_orb

            x_list.append(X)
            y_list.append(Y)
            z_list.append(Z)

        return np.array(x_list), np.array(y_list), np.array(z_list)
    
    def sma_ecc_hyperbolic_colls(self):
        vkicks = [300, 600]
        SMBH_masses = ["1e5", "4e5"]
        
        Ncoll = 0
        for kick in vkicks:
            for SMBH_Mass in SMBH_masses:
                configs_txt, configs_hdf5 = self.extract_folder(SMBH_Mass, kick)
                
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
                            print(lines[7])
                        
                        if max(mass_a, mass_b) > 1000 | units.MSun:
                            continue
                        if min(type_a, type_b) < 13:
                            continue
                        Ncoll += 1
                        
                        particle_set = read_set_from_file(hdf5, "amuse")
                        SMBH = particle_set[particle_set.mass.argmax()]
                        particle_set.position -= SMBH.position
                        particle_set.velocity -= SMBH.velocity
                        
                        coll_a = particle_set[particle_set.key == int(key1)]
                        coll_b = particle_set[particle_set.key == int(key2)]
                        
                        x_coll_a = coll_a.x.value_in(units.au)
                        y_coll_a = coll_a.y.value_in(units.au)
                        z_coll_a = coll_a.z.value_in(units.au)
                        
                        x_coll_b = coll_b.x.value_in(units.au)
                        y_coll_b = coll_b.y.value_in(units.au)
                        z_coll_b = coll_b.z.value_in(units.au)
                        
                        ax = plt.figure().add_subplot(projection='3d')
                        coll_x_arr = [ ]
                        coll_y_arr = [ ]
                        coll_z_arr = [ ]
                        for coll_ in [coll_a, coll_b]:
                            bin = SMBH + coll_
                            ke = orbital_elements(bin, G=constants.G)
                            sma = ke[2].value_in(units.au)
                            ecc = ke[3]
                            tan = ke[4].value_in(units.rad)
                            inc = ke[5].value_in(units.rad)
                            loa = ke[6].value_in(units.rad)
                            aop = ke[7].value_in(units.rad)
                            
                            x, y, z = self.make_planar(sma, ecc, tan, loa, aop, inc)
                            coll_x_arr.append(x)
                            coll_y_arr.append(y)
                            coll_z_arr.append(z)
                        
                        bin = coll_a + coll_b
                        com_particle = Particles(1)
                        com_particle.mass = bin.mass.sum()
                        com_particle.velocity = bin.center_of_mass_velocity()
                        com_particle.position = bin.center_of_mass()
                        
                        bin = Particles(particles=[SMBH, com_particle])
                        ke = orbital_elements(bin, G=constants.G)
                        sma = ke[2].value_in(units.au)
                        ecc = ke[3]
                        tan = ke[4].value_in(units.rad)
                        inc = ke[5].value_in(units.rad)
                        loa = ke[6].value_in(units.rad)
                        aop = ke[7].value_in(units.rad)
                        
                        x_c, y_c, z_c = self.make_planar(sma, ecc, tan, loa, aop, inc)
                        ax.plot(x_c, y_c, z_c, c="red", ls=":")
                        ax.plot(coll_x_arr[0], coll_y_arr[0], coll_z_arr[0], color="C0")
                        ax.plot(coll_x_arr[1], coll_y_arr[1], coll_z_arr[1], color="C0")
                        ax.scatter(x_coll_a, y_coll_a, z_coll_a, color="black", marker="X", zorder=2)
                        ax.scatter(x_coll_b, y_coll_b, z_coll_b, color="black", zorder=2)
                        ax.scatter(0, 0, 0, color="black", marker="o", s=100)
                        plt.savefig(f"plot/figures/temp/hyperbolic_orbit_{Ncoll}_3D.png", bbox_inches="tight", dpi=300)
                        plt.close()
                        
                        fig, ax = plt.subplots()
                        ax.plot(coll_x_arr[0], coll_y_arr[0], color="C0")
                        ax.plot(coll_x_arr[1], coll_y_arr[1], color="C0")
                        ax.plot(x_c, y_c, color="red", ls=":")
                        ax.scatter(x_coll_a, y_coll_a, color="black", marker="X", zorder=2)
                        ax.scatter(x_coll_b, y_coll_b, color="black", zorder=2)
                        ax.scatter(0, 0, color="black", marker="o", s=100)
                        plt.savefig(f"plot/figures/temp/hyperbolic_orbit_{Ncoll}.png", bbox_inches="tight", dpi=300)
                        plt.close()
    
    def plot_evolution(self):
        
        config = "data/hyperbolic_orbits/300kms_m1e5_config_0_merger1"
        data_files = natsort.natsorted(glob.glob(f"{config}/*.hdf5"))
        
        fig, ax = plt.subplots()
        for i, f in enumerate(data_files):
            p = read_set_from_file(f, "hdf5")
            p.move_to_center()
            print(6*constants.G*p.mass/(constants.c**2)/p.radius)
            STOP
            p.c = i
            ax.scatter(p.x.value_in(units.au), 
                       p.y.value_in(units.au), 
                       zorder=1, c="black")
            if i == 100:
                ax.scatter(p.x.value_in(units.au), 
                           p.y.value_in(units.au), 
                           marker="X", s=75, 
                           zorder=2, c="red")
                
        plt.show()     
        
       
plot = HyperbolicPlotter()
plot.sma_ecc_hyperbolic_colls()
plot.plot_evolution()