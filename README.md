This branch looks at the properties of hyper-compact stellar clusters at the instance of SMBH-SMBH merger, when the remnant experiences a kick. The code uses [AGAMA](https://github.com/GalacticDynamics-Oxford/Agama/) suite to initialise the cluster.

- To initialise your system. Execute in the root directory: ```python data/initialise_particles.py```
- To execute the N-body integrations: ```python main.py```
- To analyse the data. Execute in the root directory: ```python plot/plotter.py```

Figures are saved in ```figures/```, while data in ```data/```.