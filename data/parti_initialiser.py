import agama
from amuse.lab import new_salpeter_mass_distribution
from amuse.lab import Particles, constants, units


class MW_SMBH(object):
    """Class which defines the central SMBH"""
    def __init__(self, mass, position=[0., 0., 0.] | units.pc, velocity=[0., 0., 0.] | units.kms):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.bh_rad = (2.*constants.G*mass)/(constants.c**2.)

class ClusterInitialise(object):
    """Class to initialise the SBH particles"""
    def ZAMS_radius(self, mass):
        """
        Define stellar particle radius
        
        Args:
            mass (float):  Mass of the star
        Returns:
            radius (float):  Stellar radius
        """
        mass_sq = (mass.value_in(units.MSun))**2.
        r_zams = pow(mass.value_in(units.MSun), 1.25) \
                * (0.1148 + 0.8604*mass_sq) / (0.04651 + mass_sq)
        return r_zams | units.RSun

    def isco_radius(self, mass):
        """
        Set the SBH radius based on the Schwarzschild radius
        
        Args:
            mass (float):  Mass of the black hole
        Returns:
            radius (float):  Schwarzschild radius
        """
        return 3.*(2.*constants.G*mass)/(constants.c**2.)

    def coll_radius(self, radius):
        """
        Set the collision radius (10x the rISCO)
        
        Args:
            radius (float):  Black hole Schwarzschild radius
        Returns:
            coll_rad (float):  Collision radius
        """
        return 10.*radius

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
        nStar = 175000
        TOTAL_MASS = SMBH_parti.mass

        particles = Particles(1)
        particles[0].type = "smbh"
        particles[0].position = [0., 0., 0.] | units.pc
        particles[0].velocity = [0., 0., 0.] | units.kms
        particles[0].mass = SMBH_parti.mass
        particles[0].radius = self.isco_radius(particles[0].mass)
        particles[0].collision_radius = self.coll_radius(particles[0].radius)
        
        stars = Particles(nStar)
        masses = self.star_mass(nStar)
        
        agama.setUnits(mass=1, length=10**-3, velocity=1)
        c_pot = agama.Potential(type='Dehnen', gamma=gamma, 
                                scaleRadius=rvir.value_in(units.pc), 
                                mass=TOTAL_MASS.value_in(units.MSun))
        bh_pot = agama.Potential(type='plummer', 
                                 mass=SMBH_parti.mass.value_in(units.MSun), 
                                 scaleRadius=0)
        total_pot = agama.Potential(c_pot, bh_pot)
        c_df = agama.DistributionFunction(type='quasispherical', potential=total_pot)
        c_gm = agama.GalaxyModel(c_pot, c_df)
        
        xv, mass = c_gm.sample(nStar)
        stars.mass = masses
        stars.position = xv[:,:3] | units.pc
        stars.velocity = xv[:,3:] | units.kms
        stars.type = "star"
        stars.radius = self.ZAMS_radius(stars.mass)
        stars.collision_radius = stars.radius
        stars -= stars[stars.position.lengths() < rcavity]
        particles.add_particles(stars)
        
        particles.Nej = 0
        particles.coll_events = 0
        particles.key_tracker = particles.key
        
        return particles