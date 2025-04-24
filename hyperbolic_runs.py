from amuse.community.ph4.interface import Ph4
from amuse.community.hermite.interface import Hermite
#from amuse.community.hermitegrx.interface import HermiteGRX
from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import constants, nbody_system, units
from amuse.lab import write_set_to_file, read_set_from_file

import glob
import natsort
import numpy as np
import os


def orbital_period(sma, m1, m2):
    return np.sqrt(4 * np.pi**2 * sma**3 / (constants.G * (m1 + m2)))

def make_grav_code(particles):
    scale_mass = particles.mass.sum()
    scale_length = particles.position.lengths().max()
    
    converter = nbody_system.nbody_to_si(scale_mass, scale_length)
    gravity_code = Ph4(converter)
    gravity_code.particles.add_particles(particles)
    gravity_code.commit_particles()
    channel = gravity_code.particles.new_channel_to(particles)
    
    return gravity_code, channel

vkick = "300kms"
msmbh = "m1e5"

config = f"data/{vkick}_{msmbh}/Nimbh0_RA_BH_Run"
config = f"/media/erwanh/PhD Material/All_Data/3_Runaway_BH_At_Kick/{vkick}_{msmbh}/Nimbh0_RA_BH_Run"
colls_txt = natsort.natsorted(glob.glob(f"{config}/coll_orbital/*"))
colls_hdf5 = natsort.natsorted(glob.glob(f"{config}/merge_snapshots/*"))
for realisation in zip(colls_txt, colls_hdf5):
    txt_files = natsort.natsorted(glob.glob(f"{realisation[0]}/*.txt"))
    hdf5_files = natsort.natsorted(glob.glob(f"{realisation[1]}/*.amuse"))
    for i, (txt_df, hdf5_df) in enumerate(zip(txt_files, hdf5_files)):
        with open(txt_df, "r") as f:
            lines = f.readlines()
            key1 = int(lines[1].split("[")[-1].split("]")[0])
            key2 = int(lines[2].split("[")[-1].split("]")[0])
            mass1 = lines[3].split("[")[-1].split("]")[0] | units.MSun
            mass2 = lines[4].split("[")[-1].split("]")[0] | units.MSun
            type1 = int(lines[5].split("<")[1].split("- ")[0])
            type2 = int(lines[5].split("<")[2].split("- ")[0])
            
        if max(mass1, mass2) > 1000 | units.MSun:
            continue
        if min(type1, type2) < 10:
            continue
        
        merger_name = f"{vkick}_{msmbh}_{realisation[0].split('/')[-1]}_merger{i}"
        os.makedirs(f"data/hyperbolic_orbits/{merger_name}", exist_ok=True)
        print(f"...Simulating {merger_name}, merger {i}/{len(txt_files)}")
        
        particles = read_set_from_file(hdf5_df, "amuse")
        SMBH = particles[particles.mass.argmax()]
        coll1 = particles[particles.key == key1]
        coll2 = particles[particles.key == key2]
        
        print(coll1.radius/(6*constants.G*coll1.mass/(constants.c**2)), coll2.radius/(6*constants.G*coll2.mass/(constants.c**2)))
        """p = read_set_from_file("data/300kms_m1e5/Nimbh0_RA_BH_Run/simulation_snapshot/config_0/snapshot_step_0.amuse")
        print(coll1, coll2)
        coll1 = p[p.key == key1]
        coll2 = p[p.key == key2]
        print(coll1, coll2)
        STOP
        
        three_body_set = coll1 + coll2 + SMBH
        three_body_set.move_to_center()
        
        porb = 0 | units.s
        for p in [coll1, coll2]:
            binary = SMBH + p
            ke = orbital_elements(binary, G=constants.G)
            sma = ke[2]
            porb = max(porb, orbital_period(sma, SMBH.mass[0], p.mass[0]))
        
        time = 0 | units.s
        dt = porb/100.
        tend = 10. * porb
        print(f"Simulating till {tend.in_(units.yr)}")
        print(three_body_set.radius)
        print(6*constants.G*three_body_set.mass/(constants.c**2))
        STOP
        gravity_code, channel = make_grav_code(three_body_set)
        print("...Simulating backwards...")
        gravity_code.particles.velocity *= -1
        gravity_code.evolve_model(tend/10)
        gravity_code.particles.velocity *= -1
        channel.copy()
        gravity_code.stop()
        
        gravity_code, channel = make_grav_code(three_body_set)
        
        print("...Simulating forwards...")
        iteration = 0
        while time < tend:
            iteration += 1
            time += dt
            gravity_code.evolve_model(time)
            write_set_to_file(
                gravity_code.particles,
                f"data/hyperbolic_orbits/{merger_name}/snap_{iteration}.hdf5",
                close_file=True,
                format="hdf5",
                overwrite_file=True
            )
        gravity_code.stop()
        STOP"""
            