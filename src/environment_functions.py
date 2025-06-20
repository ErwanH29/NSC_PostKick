import os
import numpy as np

from amuse.datamodel import Particles
from amuse.units import units, constants
from amuse.ext.orbital_elements import orbital_elements


def handle_coll(bodies, colliders, tcoll, dir_path, stellar_type):
    """
    Resolve merging event
    Args:
        bodies (amuse.particles):  Simulated bodies
        colliders (amuse.particles):  The particles in the collision
        tcoll (units.time):  The time which the particles collide at
        dir_path (str):  Path to save orbital data
    Returns:
        new_particle (amuse.particle):  The new particle formed from the collision
    """
    p1 = bodies[bodies.key == colliders[0].key]
    p2 = bodies[bodies.key == colliders[1].key]

    bin_sys = Particles()
    bin_sys.add_particle(p1)
    bin_sys.add_particle(p2)
    kepler_elements = orbital_elements(bin_sys, G=constants.G)
    sem = kepler_elements[2]
    ecc = kepler_elements[3]
    inc = kepler_elements[4]
    arg_peri = kepler_elements[5]
    asc_node = kepler_elements[6]
    true_anm = kepler_elements[7]

    lines = [
        "Tcoll: {}".format(tcoll.in_(units.yr)),
        "Key1: {}".format(p1.key),
        "Key2: {}".format(p2.key),
        "M1: {}".format(p1.mass.in_(units.MSun)),
        "M2: {}".format(p2.mass.in_(units.MSun)),
        "Stellar Types: {}".format(stellar_type),
        "Semi-major axis: {}".format(abs(sem).in_(units.au)),
        "Eccentricity: {}".format(ecc),
        "Inclination: {} deg".format(inc),
        "Argument of Periapsis: {} deg".format(arg_peri),
        "Longitude of Asc. Node: {} deg".format(asc_node),
        "True Anomaly: {} deg\n".format(true_anm)
        ]

    with open(os.path.join(dir_path, f'merger_{np.sum(bodies.coll_events)}.txt'), 'w') as f:      
        for line_ in lines:
            f.write(line_)
            f.write('\n')
        f.close()

    com_pos = colliders.center_of_mass()
    com_vel = colliders.center_of_mass_velocity()

    new_particle  = Particles(1)
    new_particle.mass = colliders.total_mass()
    new_particle.collision_time = tcoll
    new_particle.position = com_pos
    new_particle.velocity = com_vel
    new_particle.coll_events = (p1.coll_events + p2.coll_events) + 1
    new_particle.stellar_type = max(stellar_type)
    bodies.remove_particles(colliders)

    return new_particle

def handle_supernova(SN_detect, bodies, grav_code):
    """
    Handle SN explosions
    Args:
        SN_detect (object): Detected particle set undergoing SN
        bodies (amuse.particles):  All bodies undergoing stellar evolution
        grav_code (code):  Gravitational integrator
    """
    SN_particle = SN_detect.particles(0)
    for ci in range(len(SN_particle)):
        SN_parti = Particles(particles=SN_particle)
        natal_kick_x = SN_parti.natal_kick_x
        natal_kick_y = SN_parti.natal_kick_y
        natal_kick_z = SN_parti.natal_kick_z

        SN_parti = SN_parti.get_intersecting_subset_in(bodies)
        SN_parti.vx += natal_kick_x
        SN_parti.vy += natal_kick_y
        SN_parti.vz += natal_kick_z
        SN_parti.SN_event = 1

    channel = bodies.new_channel_to(grav_code.particles)
    channel.copy_attributes(["vx","vy","vz"])

def ZAMS_radius(mass):
    """
    Define stellar radius
    Args:
        mass (units.mass):  Stellar mass
    Returns:
        radius (units.length):  Stellar radius
    """
    mass_sq = (mass.value_in(units.MSun))**2
    r_zams = pow(mass.value_in(units.MSun), 1.25) \
            * (0.1148 + 0.8604*mass_sq) / (0.04651 + mass_sq)
    return r_zams | units.RSun

def stellar_tidal_radius(stars, SMBH_mass):
    """
    Define stellar particle radius
    Args:
        stars (amuse.particles):  Stellar particles
        SMBH_mass (units.mass):  Mass of the SMBH
    Returns:
        radius (units.length):  Stellar particle radius
    """
    mass_sq = (stars.mass.value_in(units.MSun))**2.
    r_zams = pow(stars.mass.value_in(units.MSun), 1.25) \
                * (0.1148 + 0.8604*mass_sq) / (0.04651 + mass_sq)
    r_tidal = r_zams * (0.844**2 * SMBH_mass/stars.mass)**(1./3.)
    return r_tidal | units.RSun
  
def neutron_star_radius(mass):
    """
    Define neutron star radius
    Based on https://arxiv.org/abs/astro-ph/0002203
    Args:
        mass (units.mass):  Mass of the neutron star
    Returns:
        radius (units.length):  Neutron star radius
    """
    return 11.5 * (mass/MCH)**(-1./3.) | units.RSun

def white_dwarf_radius(mass):
    """
    Define white dwarf radius
    Based on https://arxiv.org/abs/astro-ph/0401420
    Args:
        mass (units.mass):  Mass of the neutron star
    Returns:
        radius (units.length):  Neutron star radius
    """
    return 0.0127 * (MCH/mass)**(1./3.)*(1. - (mass/MCH)**(4./3.))**(1./2.) | units.RSun

def black_hole_radius(mass):
    """
    Define black hole radius
    Args:
        mass (units.mass):  Mass of the black hole
    Returns:
        radius (units.length):  Black hole radius
    """
    return (6.*constants.G*mass)/(constants.c**2.)
  
def GW_event_kick(particles, spin_a=None, spin_b=None):
    """
    Apply recoil kick (Lousto et al. 2012)
    Args:
        particles (amuse.particles):  Particle set containing the binary
        spin_a (list):  Spin of the first particle [parallel, perpendicular]
        spin_b (list):  Spin of the second particle [parallel, perpendicular]
    Returns:
        kick (units.velocity):  Kick velocity
    """
    ### Constants:  González et al. 2007; Lousto & Zlochower 2008
    A = 1.2e4 | units.kms
    H = 6.9e3 | units.kms
    B = -0.93
    zeta = 2.53073  # Radian

    ### Constants: Lousto et al. 2012
    va1 = 3678 | units.kms
    va = 2481 | units.kms
    vb = 1793 | units.kms
    vc = 1507 | units.kms

    ke = orbital_elements(particles, G=constants.G)
    ecc = ke[3] % 1.

    if not (spin_a):
        spin_a = [0, 0]
    if not (spin_b):
        spin_b = [0, 0]

    rij = (particles[0].position - particles[1].position).value_in(units.m)
    vij = (particles[0].velocity - particles[1].velocity).value_in(units.ms)
    ang_mom = np.cross(rij, vij)

    ecc_par = ang_mom / np.linalg.norm(ang_mom)
    ecc_perp = np.cross(rij, ecc_par)
    ecc_perp /= np.linalg.norm(ecc_perp)

    q = min(particles.mass)/max(particles.mass)
    eta = q / (1. + q)**2.

    vm = A * eta**2 * np.sqrt(1. - 4.*eta) * (1.+ B*eta)
    vperp = H * eta**2 / (1 + q) * (spin_b[0] - q*spin_a[0])
    eff_spin = 2 * (spin_b[0] + q**2 * spin_a[0]) / (1+q)**2

    #vpara_a = 16 * eta**2 / (1+q) * (va1 + va*eff_spin + vb*eff_spin**2 + vc*eff_spin**3)
    #vpara_b = abs(spin_b[0] - q*spin_a[0])*np.cos(phi_delta - phi_a)
    #vpara = np.cross(vpara_a, vpara_b)

    vkick = (1. + ecc) * (vm * ecc_perp + vperp * (np.cos(zeta * ecc_perp) + np.sin(zeta * ecc_perp)) ) # + vpara*ecc_par)
    return vkick
  
MCH = 1.44 | units.MSun