
from amuse.community.seba.interface import SeBa
from amuse.lab import new_salpeter_mass_distribution
from amuse.lab import units, Particles, read_set_from_file
import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np

psets = glob.glob("300kms_m1e5/Nimbh0_RA_BH_Run/init_snapshot/*_bound.hdf5")
for p in psets:
    par = read_set_from_file(p, format='hdf5')
    print(np.unique(par.stellar_type, return_counts=True))
    

alpha = -1.5   # arXiv:0305423
mass_min = 0.5 | units.MSun   # arXiv:0305423
mass_max = 100. | units.MSun   # arXiv:1505.05473
masses = new_salpeter_mass_distribution(10000, mass_min, 
                                        mass_max, alpha
                                        ) 

stars = Particles(len(masses))
stars.mass = masses
m0 = masses.sum()
code = SeBa()
code.particles.add_particles(stars)
code.evolve_model(10 | units.Myr)
print(np.unique(code.particles.stellar_type, return_counts=True))

code.evolve_model(0.1 | units.Gyr)
print(np.unique(code.particles.stellar_type, return_counts=True))

print(code.particles.mass.sum()/m0)

code.evolve_model(1 | units.Gyr)
print(np.unique(code.particles.stellar_type, return_counts=True))

print(code.particles.mass.sum()/m0)
