import glob
import plot.LISA_Curves.LISA as li
import matplotlib.pyplot as plt
import natsort
import numpy as np
from scipy.special import jv
import statsmodels.api as sm
import pycbc.psd

from amuse.lab import units, constants
from plot.plot_class import SetupFig

M_CH = 1.44 | units.MSun

def neutron_star_radius(mass):
    """
    Define neutron star radius using https://arxiv.org/abs/astro-ph/0002203
    Args:
        mass (units.mass):  Mass of the neutron star
    Returns: Neutron star radius
    """
    return 11.5*(mass/M_CH)**(-1/3) | units.RSun

def white_dwarf_radius(mass):
    """
    Define white dwarf radius using https://arxiv.org/abs/astro-ph/0401420
    Args:
        mass (units.mass):  Mass of the neutron star
    Returns: Neutron star radius
    """
    return 0.0127 * (M_CH/mass)**(1/3) * (1 - (mass/M_CH)**(4/3))**(1/2) | units.RSun

def black_hole_radius(mass):
    """
    Define black hole radius using the Schwarzschild radius formula.
    Args:
        mass (units.mass):  Mass of the black hole
    Returns: Black hole radius
    """
    return (6.*constants.G*mass)/(constants.c**2.)


class NCSCPlotter(object):
    def __init__(self):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "cm"
        
        self.plter = SetupFig()
        self.data_labels = [
            r"$M_{\rm SMBH} = 10^{5}$ M$_\odot$", None,
            r"$M_{\rm SMBH} = 4 \times 10^{5}$ M$_\odot$", None
            ]
        self.redshift = 1
        self.lum_dist = 6801.9 | units.Mpc
        
    def extract_folder(self, SMBH_mass, vkick, folder):
        """
        Extract the data folders
        Args:
            SMBH_mass (str):  Mass of the SMBH
            vkick (str):      Velocity kick of the SMBH
            folder (str):     Folder to extract data from
        Returns: List of data folders
        """
        data_path = "/media/erwanh/PhD Material/All_Data/3_Runaway_BH_At_Kick"
        config_path = f"{vkick}kms_m{SMBH_mass}/Nimbh0_RA_BH_Run/{folder}"
        data_folders = natsort.natsorted(glob.glob(f"{data_path}/{config_path}/*"))
        return data_folders
    
    def get_sma(self, coll_rad, ecc):
        """
        Calculate the semi-major axis from the collision radius and eccentricity.
        Args:
            coll_rad (units.length):  Collision radius
            ecc (float):              Binary eccentricity
        Returns: Semi-major axis
        """
        return coll_rad/(ecc - 1)
    
    def get_impact_parameter(self, coll_rad, ecc):
        """
        Calculate the impact parameter from the collision radius and eccentricity.
        Args:
            coll_rad (units.length):  Collision radius
            ecc (float):              Binary eccentricity
        Returns: Impact parameter
        """
        return coll_rad/(np.sqrt(ecc**2 - 1))

    def get_relative_velocity(self, mass_a, mass_b, sma):
        """
        Calculate the relative velocity of the binary components.
        Args:
            mass_a (units.mass):  Mass of binary component A
            mass_b (units.mass):  Mass of binary component B
            sma (units.length):   Binary semi-major axis
        Returns:  Relative velocity
        """
        return np.sqrt(constants.G * (mass_a + mass_b)/sma)
    
    def hyperbolic_time(self, ecc, b, v0):
        """
        Calculate the hyperbolic event duration. Eqn (2) arXiv:1706.02111
        Args:
            ecc (float):          Binary eccentricity
            b (units.length):     Impact parameter
            v0 (units.velocity):  Initial velocity
        """
        numerator = 2. * (2.**(1/3) - 1.) * (ecc-1)
        denominator = ecc**2 * np.sqrt(ecc + 1.) * np.sqrt(2.**(7./6.) + (2.**(1/3)-1.)*ecc - (2.**(1./3.) +1.))
        h = numerator/denominator
        t = 0.05 * (b/(0.01 | units.au)) * (0.01 * constants.c)/v0 * h/10**-4 | units.s
        print(f"Hyperbolic event duration: {t.value_in(units.s)} s")
        return t
    
    def hyperbolic_freq(self, mass_a, mass_b, sma, ecc, rcoll) -> float:
        """
        Calculate the hyperbolic frequency (the interaction timescale).
        See: arXiv:0603441 and DOI:10.1086/155501 
        Args:
            mass_a (units.mass):   Mass of binary component A
            mass_b (units.mass):   Mass of binary component B
            sma (units.length):    Binary semi-major axis
            ecc (float):           Binary eccentricity
            rcoll (units.length):  Collision radius (assumption rperi = rcoll)
        Returns:
            freq (units.hertz):  The hyperbolic frequency
            rp (units.length):   The periastron distance
        """
        sma = self.get_sma(rcoll, ecc)
        b = self.get_impact_parameter(rcoll, ecc)
        v0 = self.get_relative_velocity(mass_a, mass_b, sma)
        
        freq = v0/(2*np.pi*b) * (ecc+1)/(ecc-1)
        t = self.hyperbolic_time(ecc, b, v0)
        return freq, t
    
    def hyperbolic_strain(self, mass_a, mass_b, ecc, rcoll) -> float:
        """
        Calculate the hyperbolic strain. 
        See arXiv:1711.09702, DOI:10.1086/155501, DOI:10.1086/156350
        Args:
            mass_a (float):  Mass of binary component A
            mass_b (float):  Mass of binary component B
            ecc (float):     Binary eccentricity
            rcoll (float):   Collision radius (assumption rperi = rcoll)
        Returns: The hyperbolic strain
        """
        #strain = (constants.G**2 * mass_a*mass_b)/(constants.c**4 * self.lum_dist * rp) # Neglect as impact parameter is coll. radius
        #b = sma*(np.sqrt(ecc**2 - 1))
        #b = coll_rad * (ecc + 1)**(3/2)/(ecc - 1)**(1/2) # arXiv:1706.02111
        #v0_sq_sq = (constants.G * (mass_a + mass_b)/b)**2 * (ecc**2 - 1)
        #v0 = np.sqrt(np.sqrt(v0_sq_sq))

        mu = (mass_a * mass_b)/(mass_a + mass_b)
        sma = self.get_sma(rcoll, ecc)
        v0 = self.get_relative_velocity(mass_a, mass_b, sma)
        gmax = 2./(ecc - 1.) * (np.sqrt(18.*(ecc + 1.) + 5. * ecc**2))
        strain = (2. * constants.G * mu * v0**2)/(self.lum_dist * constants.c**4) * gmax
        return strain
    
    def GW_freq(self, semi, nharm, mass_a, mass_b) -> float:
        """
        Frequency equation. Eqn (43) arXiv:1308.2964
        Args:
            semi (float):    Binary semi-major axis
            nharm (float):   GW harmonic mode
            mass_a (float):  Mass of binary component A
            mass_b (float):  Mass of binary component B 
        Returns: The GW frequency
        """
        freq =  np.sqrt(constants.G * (mass_a + mass_b)/abs(semi)**3.) * nharm/(2. * np.pi)
        return freq
    
    def GW_dfreq(self, semi, nharm, mass_a, mass_b, chirp_mass, ecc_func) -> float:
        """
        Account for limited LISA observation time. Eqn(6) arXiv:1811.11812 \n
        Assume Tobs ~ 5yrs.
        Args:
            semi (float):        Binary semi-major axis
            nharm (float):       GW harmonic mode
            mass_a (float):      Mass of binary component A
            mass_b (float):      Mass of binary component B
            chirp_mass (float):  Binary chirp mass
            ecc_func (float):    Value of eccentricity func. for binary
        Returns: The GW frequency rate of change
        """
        forb = np.sqrt(constants.G * (mass_a + mass_b))/(2. * np.pi) * abs(semi)**-1.5 * (self.redshift + 1)**-1.
        mass_term = (96. * nharm)/(10. * np.pi) * (constants.G * chirp_mass)**(5./3.)/(constants.c**5.)
        freq_term = (2. * np.pi * forb)**(11./3.) * abs(ecc_func)
        dfreq_dt = mass_term * freq_term
        return dfreq_dt

    def GW_strain(self, semi, ecc, freq, mass_a, mass_b, nharm) -> float:
        """
        Calculate GW strain. Eqn (7) arXiv:1811.11812 and Eqn (20) PhysRev.131.435
        Args:
            semi (float):       Binary semi-major axis
            ecc (float):        Binary eccentricity
            freq (float):       Binary GW frequency
            bin_parti (float):  Binary particle set
            mass_a (float):     Mass of binary component A
            mass_b (float):     Mass of binary component B
            nharm (float):      GW harmonic mode
        Returns: The GW strain
        """
        chirp_mass = (mass_a * mass_b)**0.6/((1 + self.redshift) * (mass_a + mass_b)**0.2)
        cfactor = 2./(3. * np.pi**(4./3.)) * (constants.G**(5./3.))/(constants.c**3. \
                 * (self.lum_dist * (1. + self.redshift))**2.)
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
        """
        Calculate the g function. Eqn (A1) PhysRev.131.435
        Args:
            ecc (float):  Binary eccentricity
            nharm (float):  GW harmonic mode
        Returns: The g function value
        """
        return nharm**4/32*((jv(nharm-2, nharm*ecc)-2*ecc*jv(nharm-1, nharm*ecc) \
               + 2/nharm*jv(nharm, nharm*ecc) + 2*ecc*jv(nharm+1, nharm*ecc) \
               - jv(nharm+2, nharm*ecc))**2 + (1-ecc**2)*(jv(nharm-2, nharm*ecc) \
               - 2*jv(nharm, nharm*ecc) + jv(nharm+2, nharm*ecc))**2 \
               + 4/(3*nharm**2)*(jv(nharm, nharm*ecc)**2))

    def GW_harmonic(self, ecc) -> float:
        """
        Peak harmonic of gravitational frequency. Eqn (36) arXiv:0211492
        Args:
            ecc (float):  Binary eccentricity
        Returns: The peak harmonic
        """ 
        return 2. * (1. + ecc)**1.1954/(1. - ecc**2.)**1.5

    def GW_time(self, semi, ecc, m1, m2) -> float:
        """
        Calculate the GW timescale. PhysRev.136.B1224
        Args:
            semi (float):   The semi-major axis of the binary
            ecc (float):    The eccentricity of the binary
            m1/m2 (float):  The binary component masses
        Returns: The gravitational wave timescale
        """
        red_mass = (m1 * m2)/(m1 + m2)
        tot_mass = m1 + m2
        coefficient = (5./256.)*(constants.c)**5./(constants.G**3.)
        tgw = coefficient*(semi**4. * (1. - ecc**2.)**3.5)/(red_mass * tot_mass**2.)
        return tgw

    def interferometer_plotter(self, ax) -> None:
        """
        Overplot interferometers on f vs. h diagram
        Use of: 
        - https://github.com/eXtremeGravityInstitute/LISA_Sensitivity/tree/master
        - https://github.com/pcampeti/SGWBProbe
        - https://pycbc.org/pycbc/latest/html/credit.html
        """
        # LISA
        lisa = li.LISA() 
        x_temp = np.linspace(1e-5, 1, 500000)
        Sn = lisa.Sn(x_temp)

        ax.plot(np.log10(x_temp), np.log10(np.sqrt(x_temp*Sn)), color='black')
        ax.text(-4.2, -19.0, 'LISA', rotation=-48, color='black',fontsize=self.plter.TICK_SIZE+3)
        
        f_lower = 10
        duration = 128
        sample_rate = 40960
        tsamples = sample_rate * duration
        fsamples = tsamples // 2 + 1
        df = 1.0 / duration
        psd = pycbc.psd.from_string('aLIGOZeroDetHighPower', fsamples, df, f_lower)
        freqs = psd.sample_frequencies
        psd = np.array(psd)
        mask = (freqs > f_lower) & np.isfinite(psd) & (psd > 0)
        
        freqs = freqs[mask]
        strain = np.sqrt(freqs * psd[mask])
        ax.plot(np.log10(freqs), np.log10(strain), color="black")
        ax.text(0.3, -22, "Adv. LIGO", fontsize=self.plter.TICK_SIZE, rotation=-43)

    def freq_strain_diagram(self):
        """Plot frequency vs. strain of GW events"""
        m4e5_300kms_colls = self.extract_folder("4e5", 300, "coll_orbital")
        m4e5_600kms_colls = self.extract_folder("4e5", 600, "coll_orbital")
        
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
        rad_array = {
            "EMRI": [],
            "NS-NS": [],
            "NS-BH": [],
            "NS-WD": [],
            "BH-BH": [],
            "BH-WD": [],
            "WD-WD": []
        }
        
        for config in [m4e5_300kms_colls, m4e5_600kms_colls]:
            for run in config:
                data_files = natsort.natsorted(glob.glob(f"{run}/*"))
                for file in data_files:
                    with open(file, 'rb') as df:
                        lines = [x.decode('utf8').strip() for x in df.readlines()]
                        mass_a = float(lines[3].split("[")[1].split("]")[0]) | units.MSun
                        mass_b = float(lines[4].split("[")[1].split("]")[0]) | units.MSun
                        type_a = int(lines[5].split("quantity<")[1].split("-")[0])
                        type_b = int(lines[5].split("quantity<")[2].split("-")[0])
                        
                        if type_a == 14:
                            coll_rad = black_hole_radius(mass_a)
                        elif type_a == 13:
                            coll_rad = neutron_star_radius(mass_a)
                        elif type_a > 9:
                            coll_rad = white_dwarf_radius(mass_a)
                        if type_b == 14:
                            coll_rad += black_hole_radius(mass_b)
                        elif type_b == 13:
                            coll_rad += neutron_star_radius(mass_b)
                        elif type_b > 9:
                            coll_rad += white_dwarf_radius(mass_b)
                        
                        if type_a > 10 and type_b > 10:
                            sma = float(lines[6].split(": ")[1][:-3]) | units.au
                            ecc = float(lines[7].split(": ")[1])
                            
                            if max(mass_a, mass_b) > 1e4 | units.MSun:
                                sma_array["EMRI"].append(sma)
                                ecc_array["EMRI"].append(ecc)
                                mass_array["EMRI"].append([mass_a, mass_b])
                                rad_array["EMRI"].append(coll_rad)
                                
                            elif type_a == 14 and type_b == 14:
                                    sma_array["BH-BH"].append(sma)
                                    ecc_array["BH-BH"].append(ecc)
                                    mass_array["BH-BH"].append([mass_a, mass_b])
                                    rad_array["BH-BH"].append(coll_rad)
                            
                            elif type_a == 13:
                                if type_b == 14:
                                    sma_array["NS-BH"].append(sma)
                                    ecc_array["NS-BH"].append(ecc)
                                    mass_array["NS-BH"].append([mass_a, mass_b])
                                    rad_array["NS-BH"].append(coll_rad)
                                elif type_b == 13:
                                    sma_array["NS-NS"].append(sma)
                                    ecc_array["NS-NS"].append(ecc)
                                    mass_array["NS-NS"].append([mass_a, mass_b])
                                    rad_array["NS-NS"].append(coll_rad)
                                else:
                                    continue  # WD with stellar-mass object is not GW event
                                    sma_array["NS-WD"].append(sma)
                                    ecc_array["NS-WD"].append(ecc)
                                    mass_array["NS-WD"].append([mass_a, mass_b])
                            
                            else:
                                continue  # WD with stellar-mass object is not GW event
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
        event_time = [ ]
        for key in sma_array.keys():
            for i in range(len(sma_array[key])):
                ecc = ecc_array[key][i]
                sma = sma_array[key][i]
                mass_a = mass_array[key][i][0]
                mass_b = mass_array[key][i][1]
                coll_rad = rad_array[key][i]
                if ecc < 1:
                    nharm = self.GW_harmonic(ecc)
                    frequency = self.GW_freq(sma, nharm, mass_a, mass_b)
                    strain = self.GW_strain(sma, ecc, frequency, mass_a, mass_b, nharm)
                else:
                    frequency, t =self.hyperbolic_freq(mass_a, mass_b, sma, ecc, coll_rad)
                    strain = self.hyperbolic_strain(mass_a, mass_b, ecc, coll_rad)  # Eqns use rp, but collision has rISCO
                    event_time.append(t)
                freq_array[key].append(frequency.value_in(units.Hz))
                strain_array[key].append(strain)
        print(f"Typical event time: {np.mean(event_time)} s, {np.median(event_time)} s")
        
        fig, ax = self.plter.get_fig_ax(figsize=(6,5))
        self.interferometer_plotter(ax)
        for i, key in enumerate(freq_array.keys()):
            x = np.log10(freq_array[key])
            y = np.log10(strain_array[key])
            if len(x) > 0:
                c = self.plter.cmap[i]
                ax.scatter(x, y, color=c)
                ax.scatter(None, None, s=50, color=c, label=key)
        ax.set_xlim(-5, 4)
        ax.legend(fontsize=12, loc="lower right")
        ax.set_xlabel(r"$\log_{10}f$ [Hz]", fontsize=self.plter.TICK_SIZE)
        ax.set_ylabel(r"$\log_{10}h$", fontsize=self.plter.TICK_SIZE)
        plt.savefig(f"plot/figures/GW_freq_strain.pdf", dpi=300, bbox_inches='tight')
        plt.close()
                            
        
plot = NCSCPlotter()
plot.freq_strain_diagram()