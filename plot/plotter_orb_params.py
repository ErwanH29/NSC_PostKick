import glob
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.ticker as mtick
from matplotlib.ticker import StrMethodFormatter
import natsort
import numpy as np
import os
import sys
from scipy.stats import gaussian_kde

from amuse.ext.orbital_elements import orbital_elements
from amuse.io.base import read_set_from_file
from amuse.lab import constants, Particles, units


def tickers(ax, ptype, sig_fig):
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
        ax.tick_params(axis="y", labelsize=TICK_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_SIZE)
        return ax
    else:
        ax.tick_params(axis="y", which='both', 
                        direction="in", 
                        labelsize=TICK_SIZE)
        ax.tick_params(axis="x", which='both', 
                        direction="in", 
                        labelsize=TICK_SIZE)
        return ax

def ecc_sma_GW(SMBH_mass):
    ecc_range = np.logspace(-3, 0, 10000)
    
    tGW = 1000 | units.yr
    avg_star_mass = 3.8 | units.MSun
    mu = SMBH_mass * avg_star_mass
    coefficient = 256*constants.G**3/(5*constants.c**5)*(mu*(SMBH_mass + avg_star_mass))*tGW
    sma_range = [((coefficient/(1-i**2)**(7./2.))**(1/4)).value_in(units.pc) for i in ecc_range]
    
    return ecc_range, sma_range

def tGW_scales(SMBH_mass, star_mass, sma, ecc):
    """
    Calculate the GW timescale based on Peters (1964).
    
    Args:
        SMBH_mass (float):  SMBH mass
        star_mass (float):  Star mass
        sma (float):  Semi-major axis
        ecc (float):  Eccentricity
    Output:
        tgw (float):  GW timescale
    """
    red_mass = (SMBH_mass * star_mass)/(SMBH_mass + star_mass)
    tot_mass = SMBH_mass + star_mass
    tgw = (5/256)*(constants.c)**5/(constants.G**3)*(sma**4*(1-ecc**2)**3.5)/(red_mass*tot_mass**2)
    
    return tgw.value_in(units.yr)

def coll_scales(SMBH_radius, star_radius, sma, ecc, distance, velocity):
    """
    Calculate the collision timescale.
    Radius of star is based on equation 1 of arXiv:1105.4966 assuming eta = 0.844.
    
    Args:
        SMBH_mass (float):  SMBH mass
        star_mass (float):  Star mass
        sma (float):  Semi-major axis
        ecc (float):  Eccentricity
    Output:
        tgw (float):  GW timescale
    """
    
    rp = sma*(1-ecc)
    coll_rad = (SMBH_radius + star_radius)
    
    if rp < coll_rad:
        time = distance/velocity
    else:
        time = np.inf | units.yr
    return time.value_in(units.yr)

def CDF_plots_sep(data_df, data_labels, fig_name, colour_array, x_label, config_label, x_lims):
    """"
    Create CDF plots for specific data
    
    Args:
        data_df (List):  Data to be plotted
        data_labels (List):  Labels for the data
        fig_name (List):  Name of the figure
        colour_array (List):  Colour array for the data
        config_label (List):  Configuration label for the data
        x_lims (List):  X-axis limits
    """

    sma_df = data_df[0]
    sep_df = data_df[1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    tickers(ax, "plot", False)
    ax.set_ylabel(r"$f_<$", fontsize=AXLABEL_SIZE)
    ax.set_xlabel(x_label, fontsize=AXLABEL_SIZE)
    for k in range(len(sma_df)):
        if k == 0 or k == 3:
            continue
        else:
            print("CDF_PLOTS_SEP", fig_name, "# data", len(sma_df[k]))
            sorted_sma = np.sort(sma_df[k])
            sma_number = [(i)/(len(sorted_sma)) for i in range(len(sorted_sma))]
            sorted_rij = np.sort(sep_df[k])
            rij_number = [(i)/len(sorted_rij) for i in range(len(sorted_rij))]    
            if len(colour_array) > 3:
                print(data_labels[k])
                ax.plot(
                    sorted_sma, sma_number, 
                    color=colour_array[k]
                )
                
                if k != 0:
                    ax.plot(
                        sorted_rij, rij_number, 
                        color=colour_array[k],
                        linestyle=":"
                    )
                
                ax.scatter(
                    None, None,
                    color=colour_array[k],
                    label=data_labels[k]
                )
            
            if len(sma_df[k]) > 0:
                with open(f"plot/figures/output/{config_label[k]}_CDF.txt", "a") as f:
                    SMBH_mass = config_label[k].split("_")[0]
                    SMBH_mass = float(SMBH_mass[1])*10**float(SMBH_mass[3]) | units.MSun
                    
                    vkick = config_label[k].split("_")[1]
                    vkick = vkick.split("V")[1]
                    vkick = vkick.split("kms")[0]
                    vkick = float(vkick) | units.kms
                    vkick = max(1 | units.kms, vkick)
                    
                    rk = constants.G*SMBH_mass/(vkick**2) 
                    distances_ratio = np.asarray([(10**i | units.pc)/rk for i in sep_df[k]])
                    ratio = (10**np.max(sep_df[k]) | units.pc)/rk
                    dratio_01_val = len(distances_ratio[distances_ratio < 0.1])/len(distances_ratio)
                    dratio_1 = len(distances_ratio[distances_ratio < 1])/len(distances_ratio)
                    
                    f.write(f"Parameter: rij\n")
                    f.write(f"SMA Mean= {10**np.mean(sma_df[k])}\n")
                    f.write(f"SMA Max= {10**np.max(sma_df[k])}\n")
                    f.write(f"SMA IQRL= {10**np.percentile(sma_df[k], 25)}\n")
                    f.write(f"SMA IQRH= {10**np.percentile(sma_df[k], 75)}\n")
                    f.write(f"rij Mean= {10**np.mean(sep_df[k])}\n")
                    f.write(f"rij Max= {10**np.max(sep_df[k])}\n")
                    f.write(f"rij IQRL= {10**np.percentile(sep_df[k], 25)}\n")
                    f.write(f"rij IQRH= {10**np.percentile(sep_df[k], 75)}\n")
                    f.write(f"max(rij)/rk = {ratio}\n")
                    f.write(f"r < 0.1rk= {dratio_01_val}\n")
                    f.write(f"r < rk= {dratio_1}\n")
                    f.write(f"======================================\n")
            
    ax.legend(fontsize=AXLABEL_SIZE-2)
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(1.e-4, 1.)
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()

def CDF_plots(data_df, data_labels, fig_name, colour_array, x_label, config_label, x_lims):
    """"
    Create CDF plots for specific data
    
    Args:
        data_df (List):  Data to be plotted
        data_labels (List):  Labels for the data
        fig_name (List):  Name of the figure
        colour_array (List):  Colour array for the data
        config_label (List):  Configuration label for the data
        x_lims (List):  X-axis limits
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    tickers(ax, "plot", False)
    ax.set_ylabel(r"$f_<$", fontsize=AXLABEL_SIZE)
    ax.set_xlabel(x_label, fontsize=AXLABEL_SIZE)
    for k in range(len(data_df)):
        print("CDF_PLOTS", fig_name, "# data", len(data_df[k]))
        if k in gamma1_idx:
            ls = ":"
        else:
            ls = "-"
        sorted_data = np.sort(data_df[k])
        if len(sorted_data) == 0:
            continue
        
        data_number = [i/(len(sorted_data)) for i in range(len(sorted_data))]
        
        if len(colour_array) > 4:
            if k != 0 and k != 3:
                ax.plot(
                    sorted_data, data_number, 
                    color=colour_array[k],
                    linestyle=ls
                )
                ax.scatter(
                    None, None,
                    color=colour_array[k],
                    label=data_labels[k]
                )
            else:  # Gamma 1 has dotted
                ax.plot(
                    sorted_data, data_number, 
                    color=colour_array[k+1], 
                    linestyle=":"
                )
        else:
            ax.plot(
                sorted_data, data_number, 
                color=colour_array[k]
            )
            ax.scatter(
                None, None,
                color=colour_array[k],
                label=data_labels[k]
            )
        
        if len(data_df[k]) > 0:
            with open(f"plot/figures/output/{config_label[k]}_CDF.txt", "a") as f:
                f.write(f"Parameter: {x_label}\n")
                f.write(f"Mean= {np.mean(data_df[k])}\n")
                f.write(f"Max= {np.max(data_df[k])}\n")
                f.write(f"IQRL= {np.mean(data_df[k]) - np.percentile(data_df[k], 25)}\n")
                f.write(f"IQRH= {np.percentile(data_df[k], 75) - np.mean(data_df[k])}\n")
                f.write(f"======================================\n")
            
    ax.legend(fontsize=AXLABEL_SIZE-2)
    ax.set_xlim(x_lims[0], x_lims[1])
    ax.set_ylim(1.e-4, 1.)
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()

def plot_system():
    """Plot the system in x, y coordinates"""
    data_file = "data/bodies_4e5_300kms_Gamma1.75.hdf5"
    crop_dist = 0.7
    initial_bodies = read_set_from_file(data_file, "hdf5")
    initial_bodies.position -= initial_bodies[initial_bodies.mass.argmax()].position
    initial_bodies -= initial_bodies[(abs(initial_bodies.x) > crop_dist | units.pc)
                                    & (abs(initial_bodies.y) > crop_dist | units.pc)]

    SMBH = initial_bodies[initial_bodies.mass.argmax()]
    minor_bodies = initial_bodies - SMBH
    for i, p in enumerate(minor_bodies):
        sys.stdout.write(f"\rProgress: {str(100*i/len(minor_bodies))[:5]}%\n")
        sys.stdout.flush()
        
        bin_sys = Particles()
        bin_sys.add_particle(p)
        bin_sys.add_particle(SMBH)
        
        ke = orbital_elements(bin_sys, G=constants.G)
        p.ecc = ke[3]

    levels = [1e-2, 1e-1, 0.5, 0.999]
    values = np.vstack([initial_bodies.x.value_in(units.pc), 
                        initial_bodies.y.value_in(units.pc)])
    xx, yy = np.mgrid[-crop_dist:crop_dist:500j, -crop_dist:crop_dist:500j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kernel = gaussian_kde(values, bw_method=0.04)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    tickers(ax, "plot", False)
    ax.set_xlim(-crop_dist, crop_dist)
    ax.set_ylim(-crop_dist, crop_dist)
    ax.set_xlabel(r"$x$ [pc]", fontsize=AXLABEL_SIZE)
    ax.set_ylabel(r"$y$ [pc]", fontsize=AXLABEL_SIZE)

    cset = ax.contour(xx, yy, f, colors="k", levels=levels, zorder=1)
    ax.scatter(initial_bodies[initial_bodies.ecc < 1].x.value_in(units.pc), 
                initial_bodies[initial_bodies.ecc < 1].y.value_in(units.pc), 
                color="C0", s=1, zorder=2, alpha=0.2)
    ax.scatter(initial_bodies[initial_bodies.mass.argmax()].x.value_in(units.pc), 
                initial_bodies[initial_bodies.mass.argmax()].y.value_in(units.pc), 
                color="black", s=5, zorder=3)

    ax.arrow(-0.6, 0.53, 0.2, 0., lw=5, head_width=0.03, fill=True, 
                facecolor="black", zorder=3)
    text = ax.text(-0.5, 0.6, r"$v_{\rm kick}$", 
                    horizontalalignment="center", 
                    fontsize=AXLABEL_SIZE+10)
    text.set_path_effects([
        path_effects.Stroke(linewidth=5, foreground='white'),
        path_effects.Normal()
    ])

    fname = "plot/figures/system_plot.png"
    plt.savefig(fname, dpi=250, bbox_inches='tight')
    plt.close()
    plt.clf()
    
def contour_plots(x_data, y_data, xlims, ylims, fname, y_label):
    """
    Create contour plots for specific data
    
    Args:
        x_data (List):  X-axis data
        y_data (List):  Y-axis data
        xlims (List):  X-axis limits
        ylims (List):  Y-axis limits
        fname (String):  Name of the figure
        y_label (String):  Y-axis label
    """
    
    values = np.vstack([x_data, y_data])
    xx, yy = np.mgrid[xlims[0]:xlims[1]:400j, ylims[0]:ylims[1]:400j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kernel = gaussian_kde(values, bw_method = "silverman")
    f = np.reshape(kernel(positions).T, xx.shape)
    f_min, f_max = np.min(f), np.max(f)
    fnorm = (f - f_min) / (f_max - f_min)
    levels = [1e-2, 1e-1, 0.5, 0.9]
    
    print("CONTOUR_PLOTS", fname, "# data", len(y_data))
        
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, fnorm, cmap="Blues", levels=levels, zorder=1, extend="max")
    cset = ax.contour(xx, yy, fnorm, colors="k", levels=levels, zorder=2)
    #if SMBH_mass is not None:
    #    ecc_data, sma_data = ecc_sma_GW(SMBH_mass)
    #    ax.plot(np.log10(sma_data), ecc_data, color="red", zorder=3)
    
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel(r"$\log_{10} a$ [pc]", fontsize=AXLABEL_SIZE)
    ax.set_ylabel(y_label, fontsize=AXLABEL_SIZE)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    tickers(ax, "hist", False)
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

def tGW_plot(data, file_name, colours, labels):
    """
    Plot the GW timescale
    
    Args:
        data (List):  Data to be plotted
        file_name (String):  Name of the figure
        colours (List):  Colour array
    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    tickers(ax, "plot", False)
    ax.set_ylabel(r"$N_<$", fontsize=AXLABEL_SIZE)
    ax.set_xlabel(r"$t_{\rm GW}$ [yr]", fontsize=AXLABEL_SIZE)
    for k in range(len(data)):
        print("TGW_PLOTS", labels[k], "# data", len(data[k]))
        if k == 0 or k == 3:
            continue
        else:
            times = np.sort(data[k])
            times = times[times < 100000]
            data_number = [i for i in range(len(times))]
            ax.plot(times, data_number, color=colours[k], label=labels[k])
    if len(data) > 0:
        ax.set_yscale("log")
        ax.legend(fontsize=AXLABEL_SIZE-2)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

def rkick_vs_rHCSC():
    vkick = [150, 300, 600, 1200]
    ratio_1e5 = [5.47267474113424, 5.617432704403266, 4.42392944505244, 2.8638758599209257]
    ratio_4e5 = [6.92616439566982, 6.199079897495372, 5.12452540921801, 3.7331061078887817]
    ratio_1e6 = [6.56620389246099, 6.420353760524145, 5.83523681059988, 4.0532758889421565]
    ratio_4e6 = [6.930077365196966, 6.536250231327551, 6.26799672111641, 5.134874958074158]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    tickers(ax, "plot", False)
    ax.set_ylabel(r"$r_{\rm HCSC}/r_{\rm kick}$", fontsize=AXLABEL_SIZE)
    ax.set_xlabel(r"$v_{\rm kick}$ [km s$^{-1}$]", fontsize=AXLABEL_SIZE)
    ax.plot(vkick, ratio_1e5, color="lightcoral", alpha=0.3)
    ax.plot(vkick, ratio_4e5, color="cornflowerblue", alpha=0.3)
    ax.plot(vkick, ratio_1e6, color="red", alpha=0.3)
    ax.plot(vkick, ratio_4e6, color="blue", alpha=0.3)
    ax.scatter(vkick, ratio_1e5, color="lightcoral", label=r"$10^{5}$ M$_\odot$", edgecolors="black")
    ax.scatter(vkick, ratio_4e5, color="cornflowerblue", label=r"$4\times10^{5}$ M$_\odot$", edgecolors="black")
    ax.scatter(vkick, ratio_1e6, color="red", label=r"$10^{6}$ M$_\odot$", edgecolors="black")
    ax.scatter(vkick, ratio_4e6, color="blue", label=r"$4\times10^{6}$ M$_\odot$", edgecolors="black")
    ax.legend(fontsize=AXLABEL_SIZE-2)
    plt.savefig("plot/figures/rHCSC_vs_rkick.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
        
CDF_PLOTS = True
SPATIAL_PLOTS = False
CONTOUR_PLOTS = False
TGW_PLOTS = True
RESET_FILES = True
PROCESS_DATA = True
np.seterr(divide='ignore')

if (RESET_FILES):
    glob.glob("plot/figures/output/*")
    for f in glob.glob("plot/figures/output/*"):
        os.remove(f)

AXLABEL_SIZE = 14
TICK_SIZE = 14

colours = ["black", "red", "blue"]
linestyle = [":", "-"]
files = [ ]

configs = natsort.natsorted(glob.glob("data/freeze_frames/*"))
configs = [i for i in configs if "hdf5" not in i]
for c in configs:
    particle_sets = natsort.natsorted(glob.glob(f"{c}/*.hdf5"))
    files = np.concatenate((files, particle_sets), axis=None)

for i, data_file in enumerate(files):
    print(f"I.C #{i}: {data_file}")
    
indices_to_remove = [2, 4, 6, 8, 10, 14, 16, 
                     18, 20, 22, 26, 28, 30, 
                     32, 34, 38, 40, 42, 44, 
                     46]
gamma1_idx = [0, 3, 7, 10, 14, 17]
files = np.delete(files, indices_to_remove)

print("..Processing...")
for i, data_file in enumerate(files):
    print(f"I.C #{i}: {data_file}")

bound_ecc_arr = [[ ] for i in range(len(files))]
bound_sma_arr = [[ ] for i in range(len(files))]
bound_inc_arr = [[ ] for i in range(len(files))]
bound_true_anom_arr = [[ ] for i in range(len(files))]
bound_arg_peri_arr = [[ ] for i in range(len(files))]
bound_sep_arr = [[ ] for i in range(len(files))]
tGW_time_arr = [[ ] for i in range(len(files))]
labels = [ ]
for i, data_file in enumerate(files):
    particles = read_set_from_file(data_file, "hdf5")
    SMBH = particles[particles.mass.argmax()]
    
    if i > 13:
        mass_parameter = "4e"+str(np.log10(SMBH.mass.value_in(units.MSun)/4))
    else:
        mass_parameter = "1e"+str(np.log10(SMBH.mass.value_in(units.MSun)))
    kick_parameter = str(abs(SMBH.vx.in_(units.kms))).split(".")[0]
    gamma_parameter = str(data_file.split("_Gamma")[-1])
    gamma_parameter = str(gamma_parameter.split("/")[0])
    
    print(f"Processing: {mass_parameter}MSun_V{kick_parameter}kms_{gamma_parameter}")
    labels.append(f"M{mass_parameter}MSun_V{kick_parameter}kms_{gamma_parameter}")
    minor_bodies = particles - SMBH
    particles.position -= SMBH.position
    particles.velocity -= SMBH.velocity
    
    for iter, p in enumerate(minor_bodies):
        sys.stdout.write(f"\rProgress: {str(100*iter/len(minor_bodies))[:5]}%")
        sys.stdout.flush()
            
        bin_sys = Particles()
        bin_sys.add_particle(p)
        bin_sys.add_particle(SMBH)
        
        ke = orbital_elements(bin_sys, G=constants.G)
        sma = ke[2]
        ecc = ke[3]
        true_anom = ke[4]
        inc = ke[5]
        arg_periapsis = ke[7]
        
        if ecc < 1:
            dist = (p.position - SMBH.position).length()
            bound_sep_arr[i].append(np.log10(dist.value_in(units.pc)))
            bound_ecc_arr[i].append(ecc)
            bound_sma_arr[i].append(np.log10(sma.value_in(units.pc)))
            bound_inc_arr[i].append(inc.value_in(units.deg))
            bound_true_anom_arr[i].append(true_anom.value_in(units.deg))
            bound_arg_peri_arr[i].append(arg_periapsis.value_in(units.deg))
            
            p.radius = p.radius*(0.844*SMBH.mass/p.mass)**(1/3)
            #tGW = tGW_scales(SMBH.mass, p.mass, sma, ecc)
            tGW = coll_scales(SMBH.radius, 
                              p.radius, 
                              sma, ecc, 
                              dist, 
                              p.velocity.length())
            tGW_time_arr[i].append(tGW)
    
    print(labels[i], "# data", len(bound_ecc_arr[i]))

data_arr = [bound_ecc_arr, 
            bound_sma_arr, 
            bound_inc_arr, 
            bound_true_anom_arr, 
            bound_arg_peri_arr]
output_labels = ["Eccentricity", 
                 "Distances", 
                 "Inclination",
                 "True Anomaly",
                 "Argument of Periapsis"]
x_labels = [r"$e$", 
            r"$\log_{10} r$ [pc]", 
            r"$i$ [$^{\circ}$]", 
            r"$\nu$ [$^{\circ}$]", 
            r"$\omega$ [$^{\circ}$]"]

# Fixed Mass: 1e5, 4e5, 1e6 | MSun
# Fixed Vkick: 75, 150, 300, 600, 1200 | km/s
cmap = matplotlib.colormaps['cool']
black_rgba = colors.to_rgba("black")
carray = np.vstack((black_rgba, black_rgba, cmap(np.linspace(0.1, 1, 6))))
colour_array = {"Fixed Mass": carray,
                "Fixed Vkick": ["lightcoral", "cornflowerblue", "red", "blue"]
                }
data_type = {"Fixed Mass": [[0, 1, 2, 3, 4, 5, 6], 
                            [14, 15, 16, 17, 18, 19, 20], 
                            [7, 8, 9, 10, 11, 12, 13],
                            [21, 22, 23, 24, 25, 26, 27],],
             "Fixed Vkick": [[2, 16, 9, 23],
                             [4, 18, 11, 25],
                             [5, 19, 12, 26],
                             [6, 20, 13, 27]]
            }
save_file = {"Fixed Mass": ["1e5MSun", "4e5MSun", "1e6MSun", "4e6MSun"],
             "Fixed Vkick": ["150kms", "300kms", "600kms", "1200kms"]
             }
data_labels = {"Fixed Mass": [None,
                              "Pre-kick", 
                              r"$150$ km s$^{-1}$",
                              None, 
                              r"$300$ km s$^{-1}$", 
                              r"$600$ km s$^{-1}$", 
                              r"$1200$ km s$^{-1}$"],
               "Fixed Vkick": [r"$10^{5}$ M$_\odot$", 
                               r"$4\times10^{5}$ M$_\odot$",
                               r"$10^{6}$ M$_\odot$",
                               r"$4\times10^{6}$ M$_\odot$",]
               }


x_lims = [[0, 1], 
          [-3, 2], 
          [0, 180], 
          [-180, 180], 
          [-180, 180]]

if (SPATIAL_PLOTS):
    print("...Plotting system...")
    plot_system()
    
if (CDF_PLOTS):
    print("...Plotting CDF...")
    
    # Plot fixing mass
    data_idx = data_type["Fixed Mass"]
    mass_par = save_file["Fixed Mass"]
    config_name = data_labels["Fixed Mass"]
    colours = colour_array["Fixed Mass"]
    
    # Iterate over masses
    for k, data_ID in enumerate(data_idx):
        for l, df_vals in enumerate(data_arr):
            config_label = [labels[i] for i in data_ID]
            fname = f"plot/figures/CDF_{mass_par[k]}_{output_labels[l]}.pdf"
            if l == 1:
                df = [df_vals[i] for i in data_ID]
                df2 = [bound_sep_arr[i] for i in data_ID]
                df = [df, df2]
                CDF_plots_sep(
                    df, config_name, fname, 
                    colours, x_labels[l], 
                    config_label,
                    x_lims[l]
                )
                
            else:
                df = [df_vals[i] for i in data_ID]
                CDF_plots(
                    df, config_name, fname, 
                    colours, x_labels[l], 
                    config_label, 
                    x_lims[l]
                )
    
    # Plot fixing kick
    data_idx = data_type["Fixed Vkick"]
    mass_par = save_file["Fixed Vkick"]
    config_name = data_labels["Fixed Vkick"]
    colours = colour_array["Fixed Vkick"]
    
    # Iterate over masses
    for k, data_ID in enumerate(data_idx):
        for l, df_vals in enumerate(data_arr):
            df = [df_vals[i] for i in data_ID]
            config_label = [labels[i] for i in data_ID]
            fname = f"plot/figures/CDF_{mass_par[k]}_{output_labels[l]}.pdf"
            
            if l == 1:
                df2 = [bound_sep_arr[i] for i in data_ID]
                df = [df, df2]
                CDF_plots_sep(
                    df, config_name, fname, 
                    colours, x_labels[l], 
                    config_label,
                    x_lims[l]
                )
                
            else:
                print(colours)
                CDF_plots(
                    df, config_name, fname, 
                    colours, x_labels[l], 
                    config_label,
                    x_lims[l]
                )
    
if (CONTOUR_PLOTS):
    for k in range(len(labels)):
        if k in gamma1_idx:
            continue
        else:
            sma_data = bound_sma_arr[k]
            ecc_data = bound_ecc_arr[k]
            inc_data = bound_inc_arr[k]
            true_anom_data = bound_true_anom_arr[k]
            
            contour_plots(
                x_data=sma_data, 
                y_data=ecc_data, 
                xlims=[-3.7, 1.4], 
                ylims=[0, 1], 
                fname=f"plot/figures/contour_{labels[k]}_sma_ecc.pdf",
                y_label=x_labels[0]
            )
            
            contour_plots(
                x_data=sma_data, 
                y_data=inc_data, 
                xlims=[-3.7, 1.4], 
                ylims=[0, 180], 
                fname=f"plot/figures/contour_{labels[k]}_sma_inc.pdf",
                y_label=x_labels[2]
            )
            
            contour_plots(
                x_data=sma_data, 
                y_data=true_anom_data, 
                xlims=[-3.7, 1.4], 
                ylims=[-180, 180], 
                fname=f"plot/figures/contour_{labels[k]}_sma_true_anom.pdf",
                y_label=x_labels[3]
            )
            
        
if (TGW_PLOTS):
    print("...Plotting tGW...")
    
    data_idx = data_type["Fixed Mass"]
    colours = colour_array["Fixed Mass"]
    config_name = data_labels["Fixed Mass"]
    for k, data_ID in enumerate(data_idx):
        print(f"Plotting {save_file['Fixed Mass'][k]}")
        if k == 1 or k == 5:
            continue
        fname = "plot/figures/tGW_" + save_file["Fixed Mass"][k] + ".pdf"
        
        df = [tGW_time_arr[i] for i in data_ID]
        tGW_plot(df, fname, colours, config_name)