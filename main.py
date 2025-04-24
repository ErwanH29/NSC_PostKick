import glob
import numpy as np
import os
import re

from amuse.io.base import read_set_from_file
from amuse.lab import units, nbody_system
from src.evol import EvolveSystem
from src.environment_functions import stellar_tidal_radius, black_hole_radius
from src.environment_functions import neutron_star_radius, white_dwarf_radius


def sort_files(path):
    sorted_files = sorted(glob.glob(path+"/*"), key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', str(x))])
    return sorted_files

def run_code(file, eta, tend, config, run_no, resume):
    """
    Function to run the code.
    
    Args:
        run_idx (Int):  Index of simulation running
        eta (Float):  Time-step parameter
        tend (Float):  Maximum simulation time
        config (String):  Configuration simulating
        run_no (Int):  Run number
        resume (Bool):  Resume simulation
    """
    no_files = 0
    if (resume):
        snapshot_files = os.path.join(config.split("config")[0], "simulation_snapshot", "config" + config.split("config")[1])
        file = (sort_files(snapshot_files))[-1]
        
    pset = read_set_from_file(file, "hdf5")
    SMBH = pset[pset.mass.argmax()]
    SMBH.stellar_type = 14 | units.stellar_type
    
    bh_mask = pset.stellar_type > 13 | units.stellar_type
    pset[bh_mask].radius = black_hole_radius(pset[bh_mask].mass)
    
    ns_mask = pset.stellar_type == 13 | units.stellar_type
    pset[ns_mask].radius = neutron_star_radius(pset[ns_mask].mass)
    
    tde_mask = pset.stellar_type < 13 | units.stellar_type
    pset[tde_mask].radius = stellar_tidal_radius(pset[tde_mask], SMBH.mass)
    
    
    fname = f"config_{run_no}"
    output_dir = config.split("config")[0]
    code_conv = nbody_system.nbody_to_si(pset.mass.sum(), 1 | units.pc)
    evolve_system = EvolveSystem(pset, tend, eta, code_conv, SMBH,
                                 no_worker=30, dir_path=output_dir,
                                 fname=fname, no_files=no_files,
                                 resume=resume)
    evolve_system.initialise_code()
    evolve_system.run_code()


vkick = "300"
mSMBH = "1e5"
Nimbh = 0
run_no = 0
suffix = "bound"

data_config = f"data/{vkick}kms_m{mSMBH}/Nimbh{Nimbh}_RA_BH_Run"
output_dir = f"{data_config}/config_{run_no}"
data_file = f"{data_config}/init_snapshot/config_{run_no}_{suffix}.hdf5"

eta = 1e-3
tend = 100 | units.kyr

run_code(file=data_file, 
         eta=eta, 
         tend=tend, 
         config=output_dir,
         run_no=run_no,
         resume=False)
