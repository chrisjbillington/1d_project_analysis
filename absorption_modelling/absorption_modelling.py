# An example of a turbulent BEC in a harmonic trap. The groundstate is found
# and then some vortices randomly printed about with a phase printing. Some
# evolution in imaginary time is then performed to smooth things out before
# evolving the BEC in time.

# Run with 'mpirun -n <N CPUs> python run_example.py'

from __future__ import division, print_function
import sys

import numpy as np

from parpde.parPDE import Simulator2D
from parpde.helmholtz_2D import Helmholtz2D


# Constants:
pi = np.pi

# Space:
nx_global = ny_global = 256
x_max_global = y_max_global = 10

simulator = Simulator2D(-x_max_global, x_max_global, -y_max_global, y_max_global, nx_global, ny_global,
                        periodic_x=True, periodic_y=True, operator_order=6)
helmholz = Helmholtz2D(simulator, use_ffts=False)

x = simulator.x
y = simulator.y

k = 2*pi

if __name__ == '__main__':
    # The initial plane wave guess:
    psi = np.exp(1j*k*x) + np.zeros_like(y)
    psi = np.array(psi, dtype=complex)

    import matplotlib.pyplot as plt
    boundary_mask = np.ones(psi.shape, dtype=bool)
    boundary_mask[0, :] = 0

    # Find the solution:
    psi = helmholz.solve(-k**2, psi, relaxation_parameter=0.5, convergence=1e-13,
                         output_interval=100, output_directory='solution', convergence_check_interval=10,
                         boundary_mask=boundary_mask)
