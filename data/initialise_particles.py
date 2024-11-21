import glob
import numpy as np
import os
import sys

from amuse.community.seba.interface import SeBa
from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import Particles, units, write_set_to_file, constants
from parti_initialiser import ClusterInitialise

def run_code(vkick, nimbh):
    """
    Run script to get initial conditions.
    
    Args:
        vkick (float):  Rough velocity of ejected SMBH
        nimbh (int):  Target # IMBH
    """
    TARGET_NSIMS = 10
    SMBH_mass = [1e5, 4e5, 1e6, 4e6] | units.MSun
    SMBH_MASS = SMBH_mass[2]
    TOTAL_IMBH_MASS = 4000 | units.MSun
    
    vdisp = 200 * (SMBH_MASS/(1.66*10**8 | units.MSun))**(1/4.86) | units.kms
    sphere_of_influence = constants.G*SMBH_MASS/vdisp**2
    rvir = 10**-2*sphere_of_influence
    rvir2 = (2*constants.G*SMBH_MASS)/(3*np.pi*vdisp**2)
    print("Virial Radius: ", rvir.in_(units.pc), rvir2.in_(units.pc))
    
    rcrop = 2*10**-4 * sphere_of_influence
    rkick = (constants.G*SMBH_MASS/vkick**2)
    
    gamma = 1.75
    TARGET_MASS = 11.7*gamma**-1.75 * SMBH_MASS \
        * (constants.G*SMBH_MASS/(sphere_of_influence*vkick**2))**(3-gamma)
    
    
    print("Configuration: "+str(nimbh))
    print(f"Target = {TARGET_MASS.in_(units.MSun)} +- {(0.05*TARGET_MASS).in_(units.MSun)}")
    data_direc = f"data/{vkick.value_in(units.kms)}kms_m{str(SMBH_MASS.value_in(units.MSun))[0]}e6/"
    config_name = "Nimbh"+str(nimbh)+"_RA_BH_Run"
    dir_path = data_direc+config_name
    nsims = len(glob.glob(dir_path+"/init_conds/*"))
    if os.path.exists(dir_path+"/coll_orbital/"):
        None
    else:
        os.mkdir(dir_path+"/")
        os.mkdir(dir_path+"/coll_orbital/")
        os.mkdir(dir_path+"/data_process/")
        os.mkdir(dir_path+"/data_process/ejec_calcs/")
        os.mkdir(dir_path+"/init_conds/")
        os.mkdir(dir_path+"/init_snapshot/")
        os.mkdir(dir_path+"/merge_snapshots/")
        os.mkdir(dir_path+"/simulation_snapshot/")
        os.mkdir(dir_path+"/simulation_stats/")
    
    while nsims < (TARGET_NSIMS):
        print("...Running simulation...")

        sbh_code = ClusterInitialise()
        pset = sbh_code.init_cluster(SMBH_MASS, rvir, gamma=gamma, rcavity=rcrop)
        pset.bound = 0
        
        SMBH = pset[pset.mass.argmax()]
        SMBH.velocity += [1,0,0] * vkick
        SMBH.bound = 1
        minor = pset - SMBH
        pset.position -= SMBH.position
        pset.velocity -= SMBH.velocity

        distances = 10*rkick
        possible = (minor.position.lengths() < distances)
        possible_particles = minor[possible]
        
        bin_sys = Particles()
        bin_sys.add_particle(SMBH)
        for i, parti_ in enumerate(possible_particles):
            sys.stdout.write(f"\rProgress: {str(100*i/len(possible_particles))[:5]}%")
            sys.stdout.flush()
            
            bin_sys.add_particle(parti_)

            kepler_elements = orbital_elements(bin_sys, G=constants.G)
            ecc = abs(kepler_elements[3])
            sma = abs(kepler_elements[2]) 
            rp = sma*(1-ecc)
            if rp < rcrop:
                pset -= parti_
            elif ecc < 1:
                parti_.bound = 1
            bin_sys.remove_particle(parti_)

        bound_stars = pset[pset.bound == 1]
        
        print("Total Particles: ", len(pset))
        print("Bounded Total mass: ", np.sum(bound_stars[bound_stars.mass != SMBH.mass].mass.in_(units.MSun)))
        print("Target mass: ", TARGET_MASS.in_(units.MSun))
        print("#Stars", len(bound_stars))
        if abs(TARGET_MASS - np.sum(bound_stars[bound_stars.mass != SMBH.mass].mass)) <= (0.1*TARGET_MASS):
            print("...successful initial conditions...")
            
            if nimbh > 0:
                imbh = bound_stars.random_sample(nimbh)
                imbh.mass = TOTAL_IMBH_MASS/nimbh
                imbh.type = "IMBH"
                imbh.radius = sbh_code.isco_radius(imbh.mass)
                imbh.collision_radius = sbh_code.coll_radius(imbh.radius)
                print("# IMBH", len(imbh))
            
            print("Cluster mass: ", pset[pset.type!="smbh"].mass.sum())
            type, counts = (np.unique(pset.type, return_counts=True))

            nsims = 0
            fname = "config_"+str(nsims)
            while os.path.exists(dir_path+"/coll_orbital/"+fname):
                print("{:} already exists".format(fname))
                nsims += 1
                fname = "config_"+str(nsims)
            
            os.mkdir(dir_path+"/coll_orbital/"+fname)
            os.mkdir(dir_path+"/merge_snapshots/"+fname)
            os.mkdir(dir_path+"/simulation_snapshot/"+fname)

            print(f"Nbound: {len(bound_stars)}")
            write_set_to_file(pset[pset.bound == 1], 
                              dir_path+"/init_snapshot/"+fname+"_bound", 
                              "hdf5", close_file=True, overwrite_file=False
                              )
            
            print(f"Ntotal: {len(pset)}")
            write_set_to_file(pset, 
                              dir_path+"/init_snapshot/"+fname+"_all", 
                              "hdf5", close_file=True, overwrite_file=False
                              )

            lines = ['Bounded Population: '+str(len(bound_stars)),
                     'Pops & Counts: '+str(type)+" "+str(counts),
                     'SMBH Mass: '+str(SMBH_MASS.in_(units.MSun)),
                     'IMBH Mass: '+str(TOTAL_IMBH_MASS.in_(units.MSun)),
                     'Stellar Mass: '+str(bound_stars[bound_stars.type=="star"].mass.sum().in_(units.MSun)),
                     'Ejection Velocity: '+str(vkick)]
            with open(os.path.join(dir_path+str("/init_conds/"), 
                'HCSC_stats_'+str(nsims)+'.txt'), 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

vkick = [150, 300, 600, 1200] | units.kms
for n in [0]:#, 2, 4, 8]:
    run_code(
        vkick=1200 | units.kms,
        nimbh=n
    )
