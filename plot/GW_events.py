import glob
import plot.LISA_Curves.LISA as li
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.ticker as mtick
import natsort
import numpy as np
from scipy.special import jv
import statsmodels.api as sm

from amuse.lab import units, constants


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
        self.redshift = 1
        self.lum_dist = 6801.9 | units.Mpc
        
        cmap = plt.colormaps['cool']
        self.cmap_colours = cmap(np.linspace(0.15, 1, 7))
        
    def extract_folder(self, SMBH_mass, vkick, folder):
        """Extract the data folders"""
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
        
    def GW_freq(self, semi, nharm, mass_a, mass_b) -> float:
        """Frequency equation. Eqn (43) arXiv:1308.2964
        
        Args:
            semi (float):  Binary semi-major axis
            nharm (float):  GW harmonic mode
            mass_a (float):  Mass of binary component A
            mass_b (float):  Mass of binary component B 
            
        Returns:
            freq (float):  The GW frequency
        """
        freq =  np.sqrt(constants.G * (mass_a + mass_b)/abs(semi)**3.) * nharm/(2. * np.pi)
        return freq
    
    def GW_dfreq(self, semi, nharm, mass_a, mass_b, chirp_mass, ecc_func) -> float:
        """Account for limited LISA observation time. Eqn(6) arXiv:1811.11812 \n
        Assume Tobs ~ 5yrs. \n
        
        Args:
            semi (float):  Binary semi-major axis
            nharm (float):  GW harmonic mode
            mass_a (float):  Mass of binary component A
            mass_b (float):  Mass of binary component B 
            chirp_mass (float):  Binary chirp mass
            ecc_func (float):  Value of eccentricity func. for binary
        Returns:
            dfreq (float):  The GW frequency rate of change
        """
        forb = np.sqrt(constants.G * (mass_a + mass_b))/(2. * np.pi) * abs(semi)**-1.5 * (self.redshift + 1)**-1.
        dfreq = (96. * nharm)/(10. * np.pi) * (constants.G * chirp_mass)**(5./3.)/(constants.c**5.) * (2. * np.pi * forb)**(11./3.) * abs(ecc_func)
        return dfreq

    def GW_strain(self, semi, ecc, freq, mass_a, mass_b, nharm) -> float:
        """Calculate GW strain. Eqn (7) arXiv:1811.11812 and Eqn (20) PhysRev.131.435 \n
        
        Args:
            semi (float):  Binary semi-major axis
            ecc (float):  Binary eccentricity
            freq (float):  Binary GW frequency
            bin_parti (float):  Binary particle set
            mass_a (float):  Mass of binary component A
            mass_b(float):  Mass of binary component B 
            nharm (float):  GW harmonic mode
        Returns:
            strain (float):  The GW strain
        """

        chirp_mass = (mass_a * mass_b)**0.6/((1 + self.redshift) * (mass_a + mass_b)**0.2)
        cfactor = 2./(3. * np.pi**(4./3.)) * (constants.G**(5./3.))/(constants.c**3. * (self.lum_dist * (1. + self.redshift))**2.)
        ecc_func = (1. + (73./24.) * ecc**2. + (37./96.) * ecc**4.)*(1. - ecc**2)**-3.5
        dfreq = self.GW_dfreq(semi, nharm, mass_a, mass_b, chirp_mass, ecc_func)
        factor = min(1., dfreq*(5. | units.yr)/freq)

        if isinstance(nharm, float):
            strain = factor * cfactor * chirp_mass**(5./3.) * freq**(-1./3.) \
                     * (2./nharm)**(2./3.) * (self.GW_gfunc(ecc, nharm)/ecc_func)
            strain = (strain.value_in(units.s**-1.6653345369377348e-16))**0.5
        else:
            strain = 0
        return strain

    def GW_gfunc(self, ecc, nharm) -> float:
        """Calculate the g function. Eqn (A1) PhysRev.131.435
        
        Args:
            ecc (float):  Binary eccentricity
            nharm (float):  GW harmonic mode
        """
        return nharm**4/32*((jv(nharm-2, nharm*ecc)-2*ecc*jv(nharm-1, nharm*ecc) \
               + 2/nharm*jv(nharm, nharm*ecc) + 2*ecc*jv(nharm+1, nharm*ecc) \
               - jv(nharm+2, nharm*ecc))**2 + (1-ecc**2)*(jv(nharm-2, nharm*ecc) \
               - 2*jv(nharm, nharm*ecc) + jv(nharm+2, nharm*ecc))**2 \
               + 4/(3*nharm**2)*(jv(nharm, nharm*ecc)**2))

    def GW_harmonic(self, ecc) -> float:
        """Peak harmonic of gravitational frequency. Eqn (36) arXiv:0211492
        
        Args:
            ecc (float):  Binary eccentricity
        Returns:
            nharm (float):  The peak harmonic
        """ 
        return 2. * (1. + ecc)**1.1954/(1. - ecc**2.)**1.5

    def GW_time(self, semi, ecc, m1, m2) -> float:
        """Calculate the GW timescale. PhysRev.136.B1224
        
        Args:
            semi (float):  The semi-major axis of the binary
            ecc (float):  The eccentricity of the binary
            m1/m2 (float):  The binary component masses
        Returns:  
            tgw (float):  The gravitational wave timescale
        """
        red_mass = (m1 * m2)/(m1 + m2)
        tot_mass = m1 + m2
        coefficient = (5./256.)*(constants.c)**5./(constants.G**3.)
        tgw = coefficient*(semi**4. * (1. - ecc**2.)**3.5)/(red_mass * tot_mass**2.)
        return tgw
        
    def freq_strain_diagram(self):
        """Plot frequency vs. strain of GW events"""
        
        m4e5_300kms_colls = self.extract_folder("4e5", 300, "coll_orbital")
        
        sma_array = {
            "EMRI": [],
            "NS-NS": [],
            "NS-BH": [],
            "NS-WD": [],
            "BH-BH": [],
            "BH-WD": [],
            "WD-WD": []
        }
        ecc_array = {
            "EMRI": [],
            "NS-NS": [],
            "NS-BH": [],
            "NS-WD": [],
            "BH-BH": [],
            "BH-WD": [],
            "WD-WD": []
        }
        mass_array = {
            "EMRI": [],
            "NS-NS": [],
            "NS-BH": [],
            "NS-WD": [],
            "BH-BH": [],
            "BH-WD": [],
            "WD-WD": []
        }
        
        N=0
        for run in m4e5_300kms_colls:
            data_files = natsort.natsorted(glob.glob(f"{run}/*"))
            for file in data_files:
                with open(file, 'rb') as df:
                    lines = [x.decode('utf8').strip() for x in df.readlines()]
                    mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                    mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun
                    type_a = int(lines[5].split("quantity<")[1].split("-")[0])
                    type_b = int(lines[5].split("quantity<")[2].split("-")[0])
                    
                    if type_a > 10 and type_b > 10:
                        sma = float(lines[6].split(": ")[1][:-3]) | units.au
                        ecc = float(lines[7].split(": ")[1])
                        print(lines[6], sma)
                        
                        if ecc < 1:
                            if type_a == 14 and type_b == 14:
                                if max(mass_a/mass_b, mass_b/mass_a) > 10**3:
                                    sma_array["EMRI"].append(sma)
                                    ecc_array["EMRI"].append(ecc)
                                    mass_array["EMRI"].append([mass_a, mass_b])
                                else:
                                    sma_array["BH-BH"].append(sma)
                                    ecc_array["BH-BH"].append(ecc)
                                    mass_array["BH-BH"].append([mass_a, mass_b])
                            
                            elif type_a == 13:
                                if type_b == 14:
                                    sma_array["NS-BH"].append(sma)
                                    ecc_array["NS-BH"].append(ecc)
                                    mass_array["NS-BH"].append([mass_a, mass_b])
                                elif type_b == 13:
                                    sma_array["NS-NS"].append(sma)
                                    ecc_array["NS-NS"].append(ecc)
                                    mass_array["NS-NS"].append([mass_a, mass_b])
                                else:
                                    sma_array["NS-WD"].append(sma)
                                    ecc_array["NS-WD"].append(ecc)
                                    mass_array["NS-WD"].append([mass_a, mass_b])
                            
                            else:
                                if type_b == 14:
                                    sma_array["BH-WD"].append(sma)
                                    ecc_array["BH-WD"].append(ecc)
                                    mass_array["BH-WD"].append([mass_a, mass_b])
                                elif type_b == 13:
                                    sma_array["NS-WD"].append(sma)
                                    ecc_array["NS-WD"].append(ecc)
                                    mass_array["NS-WD"].append([mass_a, mass_b])
                                else:
                                    sma_array["WD-WD"].append(sma)
                                    ecc_array["WD-WD"].append(ecc)
                                    mass_array["WD-WD"].append([mass_a, mass_b])
        
        freq_array = {key: [] for key in sma_array.keys()}
        strain_array = {key: [] for key in sma_array.keys()}
        for key in sma_array.keys():
            for i in range(len(sma_array[key])):
                nharm = self.GW_harmonic(ecc_array[key][i])
                frequency = self.GW_freq(sma_array[key][i], nharm, mass_array[key][i][0], mass_array[key][i][1])
                strain = self.GW_strain(sma_array[key][i], ecc_array[key][i], frequency, mass_array[key][i][0], mass_array[key][i][1], nharm)
                freq_array[key].append(frequency.value_in(units.Hz))
                strain_array[key].append(strain)
        
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4,2), height_ratios=(2,4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05
                              )
        ax = fig.add_subplot(gs[1, 0])
        ax1 = fig.add_subplot(gs[0, 0], sharex=ax)
        ax2 = fig.add_subplot(gs[1, 1], sharey=ax)
        for ax_ in [ax, ax1, ax2]:
            self.tickers(ax_, "plot", False)
            
        ax1.tick_params(axis="x", labelbottom=False)
        ax2.tick_params(axis="y", labelleft=False)
        self.interferometer_plotter(ax)
        
        for i, key in enumerate(freq_array.keys()):
            x = np.log10(freq_array[key])
            y = np.log10(strain_array[key])
            c = self.cmap_colours[i]
            if len(x) > 0:
                ax.scatter(x, y, color=c, s=15)
                ax.scatter(None, None, color=c, label=key)
                if np.sum(np.isclose(x.imag, 0)) > 10:
                    KDE_x, KDE_y = self.KDE_plotter([x,y])
                    
                    ax1.plot(KDE_x[0], KDE_x[1], color=c)
                    ax2.plot(KDE_y[0], KDE_y[1], color=c)
                    ax1.fill_between(KDE_x[0], KDE_x[1], color=c, alpha=0.35)
                    ax2.fill_between(KDE_y[0], KDE_y[1], color=c, alpha=0.35)
            
        
        ax.set_xlim(-4.6, 0)
        ax.set_ylim(-31, -17)
        ax1.set_ylim(0.01, 1.04)
        ax2.set_xlim(0.01, 1.04)
        ax.legend(
            fontsize=13, 
            bbox_to_anchor=(1.35,1.5), 
            ncol=1, frameon=False,
            borderaxespad=0.2, 
            handlelength=0.88,
            columnspacing=0.75
        )
        ax.set_xlabel(r"$\log_{10}f$ [Hz]", fontsize=self.TICK_SIZE)
        ax.set_ylabel(r"$\log_{10}h$", fontsize=self.TICK_SIZE)
        ax1.set_ylabel(r'$\rho/\rho_{\rm{max}}$', fontsize=self.TICK_SIZE)
        ax2.set_xlabel(r'$\rho/\rho_{\rm{max}}$', fontsize=self.TICK_SIZE)
        
        plt.savefig(f"plot/figures/GW_freq_strain.pdf", dpi=300, bbox_inches='tight')
        plt.close()
                
    
    def KDE_plotter(self, data) -> list:
        """Plot CDF of properties along scatter plot. arXiv:2007.04241
        
        Args:
            data (array):  The data array
            data_filt (bool):  To crop data files too large to estimate KDE
        Returns:
            data_array (list):  List hosting KDE of x data and KDE of y data
        """
            
        x = data[0]
        y = data[1]
        no_data = len(x)

        x = np.array([np.array(inner_list) for inner_list in x[:no_data]])
        kde_freq = sm.nonparametric.KDEUnivariate(x)
        kde_freq.fit()
        kde_freq.density = (kde_freq.density/max(kde_freq.density))

        y = np.array([np.array(inner_list) for inner_list in y[:no_data]])
        kde_strain = sm.nonparametric.KDEUnivariate(y)
        kde_strain.fit()
        kde_strain.density = (kde_strain.density/max(kde_strain.density))

        return [[kde_freq.support, kde_freq.density],
                [kde_strain.density, kde_strain.support]]

    def interferometer_plotter(self, ax) -> None:
        """Overplot interferometers on f vs. h diagram
        Use of: 
        - https://github.com/eXtremeGravityInstitute/LISA_Sensitivity/tree/master
        - https://github.com/pcampeti/SGWBProbe
        """
        # LISA
        lisa = li.LISA() 
        x_temp = np.linspace(1e-5, 1, 1000)
        Sn = lisa.Sn(x_temp)

        ax.plot(np.log10(x_temp), np.log10(np.sqrt(x_temp*Sn)), color='black')
        ax.text(-3.75, -19.3, 'LISA', rotation=-29, color='black',fontsize=self.TICK_SIZE+3, )
                            
        
plot = NCSCPlotter()
plot.freq_strain_diagram()