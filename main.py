import glob
import numpy as np
import re

from amuse.io.base import read_set_from_file
from amuse.lab import constants, units, nbody_system
from src.evol import EvolveSystem

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
    compact_object = pset[pset.type!="star"]
    compact_object.radius = 3*(2*constants.G*compact_object.mass)/(constants.c**2)

    fname = f"config_{run_no}"
    output_dir = config.split("config")[0]
    code_conv = nbody_system.nbody_to_si(pset.mass.sum(), 1 | units.pc)
    evolve_system = EvolveSystem(pset, tend, eta, code_conv, SMBH,
                                 no_worker=30, dir_path=output_dir,
                                 fname=fname, no_files=no_files)
    evolve_system.initialise_code()
    evolve_system.run_code()


vkick = "600"
mSMBH = "1e5"
run_no = 0

if vkick == "600" and mSMBH == "1e5":
    suffix = "all"
else:
    suffix = "bound"
data_config = f"data/{vkick}kms_m{mSMBH}"
output_dir = f"{data_config}/config_{run_no}"
data_file = f"{data_config}/init_snapshot/config_{run_no}_{suffix}"

eta = 1e-3
tend = 10 | units.kyr

run_code(file=data_file, 
         eta=eta, 
         tend=tend, 
         config=output_dir,
         run_no=run_no)

