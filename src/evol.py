import numpy as np
import os
import sys
import time as cpu_time


from amuse.community.hermite_grx.interface import HermiteGRX
from amuse.community.ph4.interface import Ph4
from amuse.community.seba.interface import SeBa
from amuse.datamodel import Particles
from amuse.lab import write_set_to_file
from amuse.units import units
from amuse.lab import constants

from src.evol_func import handle_coll, handle_supernova

class EvolveSystem(object):
    def __init__(self, parti, tend, eta, conv, 
                 GRX_set, no_worker, dir_path, 
                 fname, no_files):
        """
        Setting up the simulation code
        
        Args:
            parti (object):  The particle set needed to simulate
            tend (float):  The end time of the simulation
            eta (float):  The step size
            conv (converter):  Variable used to convert between nbody units and SI
            GRX_set (object):  SMBH particle class (only if using GRX)
            no_worker (int):  Number of workers used
            dir_path (string):  Path for outputs
            fname (string):  Template data file name
            no_files (int):  Number of snapshots already existing for called run
        """
        self.init_pset = parti
        self.tend = tend
        self.siter = 0
        print(no_files, self.tend, self.siter)
        
        self.time = 0. | units.yr
        self.eta = eta
        self.dt = self.eta * tend

        self.conv = conv
        self.SMBH = GRX_set
        self.init_time = cpu_time.time()
        self.no_workers = no_worker
        self.dpath = dir_path
        self.fname = fname
        self.coll_path = os.path.join(self.dpath, 
                                      "coll_orbital", 
                                      self.fname
                                      )
        self.no_files = no_files

    def initialise_code(self):
        """Setting up the gravitational and stellar evolution code"""
        self.particles = Particles()
        self.particles.add_particle(self.SMBH)
        self.particles.add_particle(self.init_pset[self.init_pset.mass!=self.SMBH.mass])
        self.rvir_init = self.init_pset.virial_radius()
        
        self.particles = self.init_pset
        if (1): #"Classical" in self.dpath:
            print("Classical run")
            self.grav_code = Ph4(self.conv, number_of_workers=self.no_workers)
            self.grav_code.particles.add_particles(self.particles)
            self.grav_code.parameters.timestep_parameter = 0.1
        else:
            print("PN run")
            SMBH = self.particles[self.particles.mass.argmax()]
            minor = self.particles - SMBH
            
            self.grav_code = HermiteGRX(self.conv, number_of_workers=self.no_workers)
            self.grav_code.parameters.perturbation = "1PN_Pairwise"
            self.grav_code.parameters.integrator = 'RegularizedHermite'
            self.grav_code.small_particles.add_particles(minor)
            self.grav_code.large_particles.add_particle(SMBH)
            self.grav_code.parameters.light_speed = constants.c
            self.grav_code.parameters.dt_param = 0.1

        self.grav_stopping = self.grav_code.stopping_conditions.collision_detection
        self.grav_stopping.enable()

        self.chnl_from_grav = self.grav_code.particles.new_channel_to(self.particles,
                                attributes=["mass","radius","vx","vy","vz","x","y","z"], 
                                target_names=["mass","radius","vx","vy","vz","x","y","z"]
                                )
        self.chnl_from_locl = self.particles.new_channel_to(self.grav_code.particles)
        
        self.stars = self.particles[self.particles.type == "star"]
        self.stellar_code = SeBa()
        self.stellar_code.particles.add_particles(self.stars)
        self.stellar_stopping = self.stellar_code.stopping_conditions.supernova_detection
        self.stellar_stopping.enable()
        self.star_channel = self.stellar_code.particles.new_channel_to(
                                self.grav_code.particles, 
                                attributes=["mass", "radius"], 
                                target_names=["mass", "radius"])

    def check_merger(self):
        """Check and resolve mergers"""
        for ci in range(len(self.grav_stopping.particles(0))):                
            filename = "merger_"+str(np.sum(self.particles.coll_events))+".amuse"
            self.chnl_from_grav.copy()
            
            merge_file = os.path.join(self.dpath, "merge_snapshots", 
                                      self.fname, filename
                                      )
            write_set_to_file(self.particles, merge_file, 'hdf5',
                              close_file=True, overwrite_file=True
                              )
            
            colliders = self.grav_code.stopping_conditions.collision_detection.particles
            enc_particles_set = Particles(particles=[colliders(0), colliders(1)])
            handle_coll(self.particles, 
                        enc_particles_set, 
                        self.grav_code.model_time, 
                        self.coll_path,
                        self.grav_code
                        )
            if (1):#"Classical" in self.dpath:
                self.particles.synchronize_to(self.grav_code.particles)

    def run_code(self):
        filename = f"snapshot_step_{self.siter}.amuse"
        snap_file = os.path.join(self.dpath, "simulation_snapshot", 
                                 self.fname, filename
                                 )
        
        write_set_to_file(self.particles, snap_file, 'hdf5',
                          close_file=True, overwrite_file=True
                          )
        
        while (self.time<self.tend):
            self.siter += 1
            self.time += self.dt
            self.chnl_from_locl.copy()
            
            while self.grav_code.model_time < self.time:
                self.stellar_code.evolve_model(self.time/2.)
                if self.stellar_stopping.is_set():
                    print("...Detection: SN Explosion...")
                    self.chnl_from_grav.copy()
                    handle_supernova(self.stellar_stopping, 
                                     self.stars, self.grav_code
                                     )
                self.star_channel.copy()
            
                self.grav_code.evolve_model(self.time)
                if self.grav_stopping.is_set():
                     print("........Encounter Detected........")
                     self.check_merger()
                     
                self.stellar_code.evolve_model(self.time)
                if self.stellar_stopping.is_set():
                    print("...Detection: SN Explosion...")
                    self.chnl_from_grav.copy()
                    handle_supernova(
                        self.stellar_stopping, 
                        self.stars, 
                        self.grav_code
                    )
                
                self.star_channel.copy()
                self.chnl_from_grav.copy()
                channel_from_se = self.stellar_code.particles.new_channel_to(self.particles)
                channel_from_se.copy_attributes(["luminosity"])
            
            snap_file = os.path.join(self.dpath, "simulation_snapshot", self.fname, 
                                     f"snapshot_step_{self.siter}.amuse")
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
        lines = [f'Total CPU Time: {comp_time} seconds', 
                 f'Timestep: {self.eta*self.tend}',
                 f'Simulated until: {self.time.value_in(units.yr)} years', 
                 f'Cluster Distance: {self.rvir_init.value_in(units.pc)} pc',
                 f'Total Number of Particles: {len(self.particles)}']
        with open(os.path.join(self.dpath, 'simulation_stats', self.fname+'_simulation_stats.txt'), 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
