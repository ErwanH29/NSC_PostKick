import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import StrMethodFormatter
import natsort
import numpy as np
from scipy.stats import gaussian_kde
import sys

from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import read_set_from_file, units, constants
from amuse.lab import Particle, Particles

class NCSCPlotter(object):
    def __init__(self):
        self.snapshots_300kms = natsort.natsorted(glob.glob("data/Run_300kms/simulation_snapshot/*.amuse"))
        self.snapshots_600kms = natsort.natsorted(glob.glob("data/Run_600kms/simulation_snapshot/*.amuse"))
        self.mergers_300kms = natsort.natsorted(glob.glob("data/Run_300kms/coll_orbital/*.txt"))
        self.mergers_600kms = natsort.natsorted(glob.glob("data/Run_600kms/coll_orbital/*.txt"))
        
        self.AXLABEL_SIZE = 14
        self.TICK_SIZE = 14
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        
        self.labels = [r"$300$ km s$^{-1}$", r"$600$ km s$^{-1}$"]
        
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
     
    def plot_GW_vs_time(self):
        """Plot GW events occuring in time"""
        
        BIN_RESOLUTION = 100
        TIME_PER_BIN = 100 | units.yr
        time_array = np.linspace(10**-3, 10, BIN_RESOLUTION)
        GW_events = [np.zeros(BIN_RESOLUTION), np.zeros(BIN_RESOLUTION)]
        
        for i, algorithm in enumerate([self.mergers_300kms, self.mergers_600kms]):
            for iter, file in enumerate(algorithm):
                print(file)
                with open(file, 'rb') as df:
                    lines = [x.decode('utf8').strip() for x in df.readlines()]
                    tcoll_string = lines[0].split("Tcoll")[-1]
                    tcoll = float(tcoll_string.split(" ")[1]) | units.yr
                    mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                    mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun
                    
                    if max(mass_a, mass_b) > 1000 | units.MSun:
                        idx = int(tcoll/TIME_PER_BIN)
                        GW_events[i][idx:] += 1

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(r"$t$ [kyr]", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$N_{\rm GW}$", fontsize=self.AXLABEL_SIZE)
        ax.plot(time_array, GW_events[0], 
                color="black", 
                label=self.labels[0], 
                lw=2)
        ax.plot(time_array, GW_events[1], 
                color="black", 
                label=self.labels[1], 
                lw=2, ls=":")
        ax.legend(fontsize=self.TICK_SIZE)
        ax = self.tickers(ax, "plot", True)
        ax.set_xlim(1.e-3, 10)
        ax.set_ylim(1.e-4, max(GW_events[0][-1], GW_events[1][-1]) + 1)
        plt.savefig("plot/figures/GW_vs_time.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_cluster_evolution(self):
        """Plot the half-mass radius and population evolution of the NSC and HCSC"""
        rkick = (constants.G*(4e5 | units.MSun)/(300 | units.kms)**2)
        NSC_rh_array = [[ ], [ ]]
        ejected_rh_array = [[ ], [ ]]
        ejected_pop_array = [[ ], [ ]]
        distances_array = [[ ], [ ]]
        vels = [300, 600] | units.kms
        for i, algorithm in enumerate([self.snapshots_300kms, self.snapshots_600kms]):
            ejected_r0 = None
            max_vejec = 0 | units.kms
            for j, file in enumerate(algorithm[1::20]):
                print(f"...Reading {j}/{len(algorithm[::20])}...")
                particle_set = read_set_from_file(file, "hdf5")
                SMBH = particle_set[particle_set.mass.argmax()]
                
                if j == 0:
                    distances = particle_set.position.lengths()
                    bound_HCSC = particle_set[distances < 8*rkick]
                else:  # Preserve only those who have been previously bound
                    bound_HCSC = particle_set.select(lambda x: x in bound_HCSC.key, ["key"])
                    if max(bound_HCSC.mass) < SMBH.mass:
                        bound_HCSC += SMBH
                
                bound_HCSC = self.filter_unbound(bound_HCSC)
                bound_NSC = particle_set - bound_HCSC
                
                NSC_com = Particle()
                NSC_com.mass = bound_NSC.mass.sum()
                NSC_com.position = SMBH.position
                NSC_com.x -= vels[i]*j*(10 | units.kyr/100)
                NSC_com.velocity = bound_NSC.center_of_mass_velocity()
                
                light_radii = LagrangianRadii(bound_NSC)
                vejec = 5.5 * (200 | units.kms * (SMBH.mass/(1.66*1e8 | units.MSun))**(1/4.87))
                ejected_vels = (bound_NSC.velocity - NSC_com.velocity).lengths() > vejec
                ejected_dist = (bound_NSC.position - NSC_com.position).lengths() > light_radii[8]  #90th percentile
                possible_ejectees = ejected_vels & ejected_dist
                ejected_stars = bound_NSC[possible_ejectees]
                for p in ejected_stars:
                    rij = (p.position - NSC_com.position)
                    vij = (p.velocity - NSC_com.velocity)
                    
                    trajectory = (np.dot(rij, vij))/(rij.length()*vij.length())
                    if trajectory < 0:
                        ejected_stars -= p
                    
                bound_NSC -= ejected_stars
                if len(ejected_stars) > 2:
                    max_vejec = max(max_vejec, ejected_stars.velocity.lengths().max())
                    light_radii = LagrangianRadii(ejected_stars)
                    if ejected_r0 is None:
                        ejected_r0 = light_radii[6]
                    ratio = light_radii[6]/ejected_r0
                else:
                    ratio = np.inf
                        
                print(f"\nBound HCSC: {len(bound_HCSC)}", end=" ")
                print(f"Bound NSC: {len(bound_NSC)}", end=" ")
                print(f"Ejected Stars: {len(ejected_stars)}", end=" ")
                print(f"Max HCSC mass: {bound_HCSC.mass.max().in_(units.MSun)}") 
                    
                light_radii = LagrangianRadii(bound_NSC)
                if j == 0:
                    NSC_r0 = light_radii[6]
                ratio = light_radii[6]/NSC_r0
                NSC_rh_array[i].append(ratio)
                
                ejected_rh_array[i].append(ratio)
                ejected_pop_array[i].append(len(ejected_stars)/len(particle_set))
                if j == 0 or j == len(algorithm[1::20]) - 1:
                    distances = bound_NSC.position.lengths().value_in(units.pc)
                    distances_array[i].append([distances])
            print(f"Max velocity of ejected stars: {max_vejec.in_(units.kms)}")
            
        time_array0 = [0.1*i for i in range(len(NSC_rh_array[0]))]
        time_array1 = [0.1*i for i in range(len(NSC_rh_array[1]))]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(r"$t$ [kyr]", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$r_{\rm half}/r_{\rm half, 0}$", 
                      color="red",
                      fontsize=self.AXLABEL_SIZE)
        ax.plot(time_array0, NSC_rh_array[0], 
                color="red", 
                label=r"$300$ km s$^{-1}$", 
                lw=2)
        ax.plot(time_array1, NSC_rh_array[1], 
                color="red", 
                label=r"$600$ km s$^{-1}$", 
                lw=2, ls=":")
        ax.tick_params(axis="y", labelcolor="red", 
                       direction='in', right=False, 
                       labelsize=self.TICK_SIZE)
        
        ax2 = ax.twinx()
        ax2.plot(time_array0, ejected_pop_array[0], color="blue")
        ax2.plot(time_array1, ejected_pop_array[1], color="blue", ls=":")
        ax2.set_ylabel(r"$N_{\rm ejected}/N_{\rm NSCS, 0}$", color="blue", fontsize=self.TICK_SIZE)
        ax2.tick_params(axis="y", labelcolor="blue", direction='in', labelsize=self.TICK_SIZE)
                
        ax.xaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax2.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        ax.legend(fontsize=self.TICK_SIZE)
        #ax = self.tickers(ax, "plot", False)
        ax.set_xlim(1.e-3, 10)
        plt.savefig("plot/figures/halfmass_NSC.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        data_array = [distances_array[0][0], 
                      distances_array[1][0], 
                      distances_array[0][1], 
                      distances_array[1][1]]
        colours = ["red", "blue", "lightcoral", "cornflowerblue"]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$r$ [pc]", fontsize=self.AXLABEL_SIZE)
        ax.set_ylabel(r"$\rho(r)/\rho_0$", fontsize=self.AXLABEL_SIZE)
        for i, data in enumerate(data_array):
            print(data[0])
            print("==================")
            kde_values = gaussian_kde(data[0])
            rij_range = np.linspace(min(data[0]), max(data[0]), 1000)
            pdf_values = kde_values(rij_range)
            pdf_values = pdf_values/pdf_values.max()
            
            ax.plot(rij_range, pdf_values, color=colours[i], lw=2)
        ax.set_xlim(1.e-3, 10)
        ax.scatter(None, None, color="red", label=r"$300$ km s$^{-1}$")
        ax.scatter(None, None, color="blue", label=r"$600$ km s$^{-1}$")
        ax.legend(fontsize=self.TICK_SIZE)
        plt.savefig("plot/figures/density_distr.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        
plot = NCSCPlotter()
plot.plot_GW_vs_time()
plot.plot_cluster_evolution()