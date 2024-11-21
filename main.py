import glob
import numpy as np
import re

from amuse.io.base import read_set_from_file
from amuse.lab import constants, units, nbody_system
from src.evol import EvolveSystem


def stellar_tidal_radius(stars, SMBH_mass):
    """
    Define stellar particle radius using Eqn 1 of arXiv:1105.4966
        
    Args:
        stars (object):  Stellar particles
        SMBH_mass (float):  Mass of the SMBH
    Returns:
        radius (float):  Stellar particle radius
    """
    mass_sq = (stars.mass.value_in(units.MSun))**2.
    r_zams = pow(stars.mass.value_in(units.MSun), 1.25) \
                * (0.1148 + 0.8604*mass_sq) / (0.04651 + mass_sq)
    radius = r_zams * (0.844**2 * SMBH_mass/stars.mass)**(1./3.)
    return radius | units.RSun

def sort_files(path):
    sorted_files = sorted(glob.glob(path+"/*"), key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', str(x))])
    return sorted_files

def run_code(file, eta, tend, config, run_no):
    """
    Function to run the code.
    
    Args:
        run_idx (Int):  Index of simulation running
        eta (Float):  Time-step parameter
        tend (Float):  Maximum simulation time
        config (String):  Configuration simulating
        run_no (Int):  Run number
    """
    no_files = 0

    pset = read_set_from_file(file, "hdf5")
    SMBH = pset[pset.mass == pset.mass.max()]
    SMBH.stellar_type = 14 | units.stellar_type
 
    compact_objects = pset.stellar_type > 13 | units.stellar_type
    pset[compact_objects].radius = 3*(2*constants.G*pset[compact_objects].mass)/(constants.c**2)
    pset[~compact_objects].radius = stellar_tidal_radius(pset[~compact_objects], SMBH.mass)
    
    fname = f"config_{run_no}"
    output_dir = config.split("config")[0]
    code_conv = nbody_system.nbody_to_si(pset.mass.sum(), 1 | units.pc)
    evolve_system = EvolveSystem(pset, tend, eta, code_conv, SMBH,
                                 no_worker=4, dir_path=output_dir,
                                 fname=fname, no_files=no_files)
    evolve_system.initialise_code()
    evolve_system.run_code()


vkick = "300"
mSMBH = "1e5"
Nimbh = 0
run_no = 9 
suffix = "bound"

data_config = f"data/{vkick}kms_m{mSMBH}/Nimbh{Nimbh}_RA_BH_Run"
output_dir = f"{data_config}/config_{run_no}"
data_file = f"{data_config}/init_snapshot/config_{run_no}_{suffix}"

eta = 1e-3
tend = 10 | units.kyr

run_code(file=data_file, 
         eta=eta, 
         tend=tend, 
         config=output_dir,
         run_no=run_no)

