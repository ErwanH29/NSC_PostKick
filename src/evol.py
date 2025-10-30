import glob
import numpy as np
import os
import time as cpu_time

from amuse.community.ph4.interface import Ph4
from amuse.community.seba.interface import SeBa
from amuse.datamodel import Particles
from amuse.lab import write_set_to_file
from amuse.units import units

from src.environment_functions import handle_coll, handle_supernova
from src.environment_functions import stellar_tidal_radius, white_dwarf_radius
from src.environment_functions import GW_event_kick, black_hole_radius, neutron_star_radius

class EvolveSystem(object):
    def __init__(self, bodies, tend, eta, conv, no_worker, 
                 dir_path, fname, no_files, resume=False):
        """
        Setting up the simulation code
        Args:
            bodies (object):    The particle set needed to simulate
            tend (units.time):  The end time of the simulation
            eta (float):        The step size
            conv (converter):   Variable used to convert between nbody units and SI
            no_worker (int):    Number of workers used
            dir_path (string):  Path for outputs
            fname (string):     Template data file name
            no_files (int):     Number of snapshots already existing for called run
        """
        self.init_pset = bodies
        self.tend = tend
        self.siter = 0
        self.resume_time = 0 | units.yr
        if (resume):
            no_files = glob.glob(os.path.join(dir_path, "simulation_snapshot", fname, "*"))
            self.siter = len(no_files) - 1
            self.tend = self.tend - (0.1 * self.siter * eta | units.Myr)
            self.resume_time = (0.1 * self.siter * eta | units.Myr)

        self.time = 0. | units.yr
        self.eta = eta
        self.dt = self.eta * tend

        self.conv = conv
        self.init_time = cpu_time.time()
        self.no_workers = no_worker
        self.dpath = dir_path
        self.fname = fname
        self.coll_path = os.path.join(self.dpath, "coll_orbital", self.fname)
        self.no_files = no_files

    def initialise_code(self):
        """Setting up the gravitational and stellar evolution code"""
        SMBH = self.init_pset[self.init_pset.mass.argmax()]
        minor_bodies = self.init_pset - SMBH
        
        self.particles = Particles()
        self.particles.add_particle(SMBH)
        self.particles.add_particle(minor_bodies)
        self.rvir_init = self.init_pset.virial_radius()
        
        self.particles = self.init_pset
        self.grav_code = Ph4(self.conv, number_of_workers=self.no_workers)
        self.grav_code.particles.add_particles(self.particles)
        self.grav_code.parameters.timestep_parameter = 0.03

        self.grav_stopping = self.grav_code.stopping_conditions.collision_detection
        self.grav_stopping.enable()

        self.chnl_from_grav = self.grav_code.particles.new_channel_to(
            self.particles,
            attributes=["mass","vx","vy","vz","x","y","z"], 
            target_names=["mass","vx","vy","vz","x","y","z"]
            )
        self.chnl_from_locl = self.particles.new_channel_to(self.grav_code.particles)

        # Only evolve stars
        self.stars = self.particles[self.particles.stellar_type < (10 | units.stellar_type)]
        self.stars.relative_mass = self.stars.mass
        self.stars.relative_age = self.resume_time + (100. | units.Myr)

        self.stellar_code = SeBa()
        self.stellar_code.particles.add_particles(self.stars)
        self.stellar_stopping = self.stellar_code.stopping_conditions.supernova_detection
        self.stellar_stopping.enable()

        self.star_grav_channel = self.stellar_code.particles.new_channel_to(
            self.grav_code.particles, 
            attributes=["mass"], 
            target_names=["mass"]
            )
        self.star_local_channel = self.stellar_code.particles.new_channel_to(
            self.particles, 
            attributes=["stellar_type"], 
            target_names=["stellar_type"]
            )

    def process_merger(self, enc_particles_set, stellar_type_array):
        """
        Process the merger of two colliding particles
        Args:
            enc_particles_set (Particles):  Set of colliding particles
            stellar_type_array (list):      Stellar type of colliding particles
        Returns: Remnant particle
        """
        filename = "merger_"+str(np.sum(self.particles.coll_events))+".amuse"
        merge_file = os.path.join(self.dpath, "merge_snapshots", self.fname, filename)
        tcoll = self.grav_code.model_time + self.resume_time

        write_set_to_file(
            self.particles, 
            merge_file, 'hdf5',
            close_file=True, 
            overwrite_file=True
            )

        newp = handle_coll(
            self.particles, 
            enc_particles_set, 
            tcoll, 
            self.coll_path,
            stellar_type_array
            )
        
        return newp

    def check_merger(self):
        """Check and resolve mergers"""
        stellar_particles = self.stellar_code.particles
        for ci in range(len(self.grav_stopping.particles(0))): 
            print(f"...Detection: Collision {ci+1}...")               
            self.chnl_from_grav.copy()
            Ngrav = len(self.grav_code.particles)

            colliders = self.grav_stopping.particles
            enc_particles_set = Particles(particles=[colliders(0), colliders(1)])
            stellar_type_arr = [ ]
            for p in enc_particles_set:
               collider = p.as_particle_in_set(self.particles)
               stellar_type_arr.append(collider.stellar_type)

            SMBH = self.particles[self.particles.mass.argmax()]
            distance = (enc_particles_set[0].position - enc_particles_set[1].position).length()
            if max(stellar_type_arr) < 13 | units.stellar_type:    # Non-TDE collision
                if stellar_type_arr[0] < 10 | units.stellar_type:  # Star --> use SeBa radius
                    coll_a_radius = enc_particles_set[0].as_particle_in_set(stellar_particles).radius
                else:  # Scale down collider to white dwarf radius
                    coll_a_radius = white_dwarf_radius(enc_particles_set[0].mass)

                if stellar_type_arr[1] < 10 | units.stellar_type:
                    coll_b_radius = enc_particles_set[1].as_particle_in_set(stellar_particles).radius
                else:
                    coll_b_radius = white_dwarf_radius(enc_particles_set[1].mass)

                # New star radius becomes SeBa given radius, or tidal radius
                if distance <= (coll_a_radius + coll_b_radius):
                    print("Star-Star collision")
                    newp = self.process_merger(enc_particles_set, stellar_type_arr)
                    stellar_particles.add_particle(newp)
                    newp.radius = stellar_tidal_radius(newp, self.particles.mass.max())
                    self.grav_code.particles.add_particle(newp)

            elif min(stellar_type_arr) < 13 | units.stellar_type:  # Compact object - Star
                if max(enc_particles_set.mass) < 0.75*SMBH.mass \
                    and min(stellar_type_arr) < 10 | units.stellar_type:  # Non-SMBH TDE
                    st_idx = np.asarray(stellar_type_arr).argmin()
                    co_idx = np.asarray(stellar_type_arr).argmax()

                    star = enc_particles_set[st_idx]
                    radius = star.as_particle_in_set(stellar_particles).radius
                    radius *= (enc_particles_set[co_idx].mass/SMBH.mass)**(1./3.)
                    if distance < (radius + enc_particles_set[co_idx].radius):
                        newp = self.process_merger(enc_particles_set, stellar_type_arr)
                        if max (stellar_type_arr) == 13 | units.stellar_type:
                            print("NS - Star TDE")
                            newp.radius = neutron_star_radius(newp.mass)
                        else:
                            print("BH - Star TDE")
                            newp.radius = black_hole_radius(newp.mass)

                        self.grav_code.particles.add_particle(newp)
                        
                elif max(enc_particles_set.mass) < 0.75*SMBH.mass \
                    and min(stellar_type_arr) >= 10 | units.stellar_type:
                    st_idx = np.asarray(stellar_type_arr).argmin()
                    co_idx = np.asarray(stellar_type_arr).argmax()
                    
                    white_dwarf = enc_particles_set[st_idx]
                    radius = white_dwarf_radius(white_dwarf.mass)
                    if distance < (radius + enc_particles_set[co_idx].radius):
                        newp = self.process_merger(enc_particles_set, stellar_type_arr)
                        if stellar_type_arr[st_idx] == 13 | units.stellar_type:
                            print("WD - NS TDE")
                            newp.radius = neutron_star_radius(newp.mass)
                        else:
                            print("WD - BH TDE")
                            newp.radius = black_hole_radius(newp.mass)

                        self.grav_code.particles.add_particle(newp)

                else:  # SMBH TDE
                    print("SMBH TDE")
                    newp = self.process_merger(enc_particles_set, stellar_type_arr)
                    newp.radius = black_hole_radius(newp.mass)
                    self.grav_code.particles.add_particle(newp)

            elif max(stellar_type_arr) == 13 | units.stellar_type:  # NS - NS merger
                print("NS - NS merger")
                newp = self.process_merger(enc_particles_set, stellar_type_arr)
                newp.radius = neutron_star_radius(newp.mass)
                self.grav_code.particles.add_particle(newp)

            else:  # GW event
                print("GW Event")
                newp = self.process_merger(enc_particles_set, stellar_type_arr)
                newp.radius = black_hole_radius(newp.mass)

                bin_sys = enc_particles_set.copy()
                bin_sys.move_to_center()
                recoil_kick = GW_event_kick(bin_sys)
                newp.velocity += recoil_kick

                print(f"Applied vkick: {recoil_kick.in_(units.kms)}")
                self.grav_code.particles.add_particle(newp)

            Ngrav_post = len(self.grav_code.particles)
            if Ngrav_post != Ngrav:
                coll_a = enc_particles_set[0].as_particle_in_set(stellar_particles)
                coll_b = enc_particles_set[1].as_particle_in_set(stellar_particles)
                if coll_a is not None:
                    stellar_particles.remove_particle(coll_a)
                if coll_b is not None:
                    stellar_particles.remove_particle(coll_b)
                self.grav_code.particles.remove_particle(colliders(1))
                self.grav_code.particles.remove_particle(colliders(0))
                self.particles.add_particle(newp)

    def run_code(self):
        """Run the simulation code"""
        filename = f"snapshot_step_{self.siter}.amuse"
        snap_file = os.path.join(self.dpath, "simulation_snapshot", self.fname, filename)
        write_set_to_file(
            self.particles, 
            snap_file, 'hdf5',
            close_file=True, 
            overwrite_file=True
        )
        
        while (self.time < self.tend):
            self.siter += 1
            self.time += self.dt
            self.chnl_from_locl.copy()

            while self.grav_code.model_time < self.time:
                self.stellar_code.evolve_model(self.time/2.)

                if self.stellar_stopping.is_set():
                    print("...Detection: SN Explosion...")
                    self.chnl_from_grav.copy()
                    handle_supernova(
                        self.stellar_stopping, 
                        self.stars, self.grav_code
                        )
                self.star_grav_channel.copy()

                self.grav_code.evolve_model(self.time)
                if self.grav_stopping.is_set():
                    self.star_local_channel.copy()
                    self.check_merger()

                self.chnl_from_grav.copy()

                self.stellar_code.evolve_model(self.time)
                if self.stellar_stopping.is_set():
                    print("...Detection: SN Explosion...")
                    self.chnl_from_grav.copy()
                    handle_supernova(
                        self.stellar_stopping, 
                        self.stars, 
                        self.grav_code
                    )

            snap_file = os.path.join(
                self.dpath, "simulation_snapshot", self.fname, 
                f"snapshot_step_{self.siter}.amuse"
                )
            write_set_to_file(
                self.particles, 
                snap_file, 
                'hdf5', 
                close_file=True, 
                overwrite_file=True
                )

        self.grav_code.stop()
        self.stellar_code.stop()
        comp_end = cpu_time.time()   
        comp_time = comp_end - self.init_time
        
        print('...Saving Data...')
        lines = [
            f'Total CPU Time: {comp_time} seconds', 
            f'Timestep: {self.eta*self.tend}',
            f'Simulated until: {self.time.value_in(units.yr)} years', 
            f'Cluster Distance: {self.rvir_init.value_in(units.pc)} pc',
            f'Total Number of Particles: {len(self.particles)}'
            ]
        with open(os.path.join(self.dpath, 'simulation_stats', self.fname+'_simulation_stats.txt'), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')