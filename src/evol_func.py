import os
import numpy as np

from amuse.datamodel import Particles
from amuse.units import units, constants
from amuse.ext.orbital_elements import orbital_elements_from_binary


def handle_coll(parti, parti_in_enc, tcoll, dir_path, code):
    """
    Resolve merging event
    
    Args:
      parti (object):  The complete particle set being simulated
      parti_in_enc (object):  The particles in the collision
      tcoll (float):  The time which the particles collide at
      code (code):  The integrator used
      dir_path (string):  Path to save orbital data
    Returns:
      new_particle (object):  The new particle formed from the collision
    """
    p1 = parti[parti.key == parti_in_enc[0].key]
    p2 = parti[parti.key == parti_in_enc[1].key]
    if p1.type == "smbh" or p2.type == "smbh":
       new_type="smbh"
    elif p1.type == "star" and p2.type == "star":
       new_type = "star"    
    else:
       new_type = "IMBH"

    bin_sys = Particles()
    bin_sys.add_particle(p1)
    bin_sys.add_particle(p2)
    kepler_elements = orbital_elements_from_binary(bin_sys, G=constants.G)
    sem = kepler_elements[2]
    ecc = kepler_elements[3]
    inc = kepler_elements[4]
    arg_peri = kepler_elements[5]
    asc_node = kepler_elements[6]
    true_anm = kepler_elements[7]

    lines = ["Tcoll: {}".format(tcoll.in_(units.yr)),
             "Key1: {}".format(p1.key),
             "Key2: {}".format(p2.key),
             "M1: {}".format(p1.mass.in_(units.MSun)),
             "M2: {}".format(p2.mass.in_(units.MSun)),
             "Semi-major axis: {}".format(abs(sem).in_(units.au)),
             "Eccentricity: {}".format(ecc),
             "Inclination: {} deg".format(inc),
             "Argument of Periapsis: {} deg".format(arg_peri),
             "Longitude of Asc. Node: {} deg".format(asc_node),
             "True Anomaly: {} deg\n".format(true_anm)
            ]

    with open(os.path.join(dir_path, f'merger_{np.sum(parti.coll_events)}.txt'), 'w') as f:      
        for line_ in lines:
            f.write(line_)
            f.write('\n')
        f.close()

    com_pos = parti_in_enc.center_of_mass()
    com_vel = parti_in_enc.center_of_mass_velocity()

    new_particle  = Particles(1)
    new_particle.mass = parti_in_enc.total_mass()
    new_particle.collision_time = tcoll
    new_particle.position = com_pos
    new_particle.velocity = com_vel
    new_particle.coll_events = (p1.coll_events + p2.coll_events) + 1
    new_particle.type = new_type
    if max(parti_in_enc.mass) > 125 | units.MSun:
       new_particle.radius = 10.*(6.*constants.G*new_particle.mass)/(constants.c**2.)  # 10x rISCO
    else:
       new_particle.radius = ZAMS_radius(new_particle.mass)
    new_particle.bound = max(parti_in_enc.bound)
    
    if (0):#"Classical" not in dir_path:
        if new_particle.mass > 1000 | units.MSun:
            code.particles.remove_particles(parti_in_enc)
            code.large_particles.add_particles(new_particle)
        else:
            code.particles.remove_particles(parti_in_enc)
            code.small_particles.add_particles(new_particle)
    
    parti.add_particles(new_particle)
    parti.remove_particles(parti_in_enc)
    
    return new_particle

def handle_supernova(SN_detect, bodies, grav_code):
    """
    Function handling SN explosions
    
    Args:
      SN_detect (object): Detected particle set undergoing SN
      bodies (object):  All bodies undergoing stellar evolution
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
    Define particle radius
    
    Args:
      mass (float): Stellar mass
    Returns:
      radius (float): Stellar radius
    """
    mass_sq = (mass.value_in(units.MSun))**2
    r_zams = pow(mass.value_in(units.MSun), 1.25) \
            * (0.1148 + 0.8604*mass_sq) / (0.04651 + mass_sq)
    return r_zams | units.RSun
