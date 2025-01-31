import agama

from amuse.community.seba.interface import SeBa
from amuse.lab import new_salpeter_mass_distribution
from amuse.lab import Particles, constants, units


class MW_SMBH(object):
    """Class which defines the central SMBH"""
    def __init__(self, 
                 mass, 
                 position=[0., 0., 0.] | units.pc, 
                 velocity=[0., 0., 0.] | units.kms):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.bh_rad = (2.*constants.G*mass)/(constants.c**2.)

class ClusterInitialise(object):
    """Class to initialise the SBH particles"""
    def star_mass(self, nStar):
        """
        Set stellar particle masses
        
        Args:
            nStar (int):  Number of stellar particles
        Returns:
            masses (float):  Mass distribution for stellar particles
        """
        alpha = -1.5   # arXiv:0305423
        mass_min = 0.5 | units.MSun   # arXiv:0305423
        mass_max = 100. | units.MSun   # arXiv:1505.05473
        return new_salpeter_mass_distribution(nStar, mass_min, 
                                              mass_max, alpha
                                              ) 

    def init_cluster(self, mass, rvir, gamma, rcavity):
        """
        Initialise the cluster. 
        Makes use of the AGAMA framework (https://github.com/GalacticDynamics-Oxford/Agama/)
        
        Args:
            mass (float):  Mass of SMBH
            rvir (float):  Cluster initial virial radius
            gamma (float):  Cluster density power-law
            rcavity (float):  Cluster cavity size
        Returns:
            particles (object):  Particle set
        """
        SMBH_parti = MW_SMBH(mass)
        nStar = 25000
        print(f"#Stars ={nStar}")

        particles = Particles(1)
        particles[0].type = "smbh"
        particles[0].position = [0., 0., 0.] | units.pc
        particles[0].velocity = [0., 0., 0.] | units.kms
        particles[0].mass = SMBH_parti.mass
        
        stars = Particles(nStar)
        masses = self.star_mass(nStar)
        stars.mass = masses
        stellar_code = SeBa()
        stellar_code.particles.add_particle(stars)
        stellar_code.evolve_model(0.1 | units.Gyr)
        channel = stellar_code.particles.new_channel_to(stars)
        channel.copy_attributes(["mass", "stellar_type"])
        stellar_code.stop()
        
        cluster_mass = stars.mass.sum() + SMBH_parti.mass
        
        agama.setUnits(mass=1, length=10**-3, velocity=1)
        c_pot = agama.Potential(type='Dehnen', gamma=gamma, 
                                scaleRadius=rvir.value_in(units.pc), 
                                mass=cluster_mass.value_in(units.MSun))
        bh_pot = agama.Potential(type='plummer', 
                                 mass=SMBH_parti.mass.value_in(units.MSun), 
                                 scaleRadius=SMBH_parti.bh_rad.value_in(units.pc))
        total_pot = agama.Potential(c_pot, bh_pot)
        c_df = agama.DistributionFunction(type='quasispherical', potential=total_pot)
        c_gm = agama.GalaxyModel(total_pot, c_df)
        xv, _ = c_gm.sample(nStar)
        
        stars.position = xv[:,:3] | units.pc
        stars.velocity = xv[:,3:] | units.kms
        stars -= stars[stars.position.lengths() < rcavity]
        
        particles.add_particles(stars)
        
        particles.Nej = 0
        particles.coll_events = 0
        print(f"Trimmed off: {nStar - len(particles)}")
        
        return particles