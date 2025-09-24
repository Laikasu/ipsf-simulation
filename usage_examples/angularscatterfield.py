import numpy as np
import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt



params = DesignParams(scat_mat="gold", diameter=100)
capture_angle_medium = np.arcsin(min(params.NA/params.n_medium, 1))
angles = np.linspace(np.pi, np.pi-capture_angle_medium, 100)
scatter_field = calculate_scatter_field(params)

intensity = np.sum(np.abs(scatter_field), axis=-1)
phases = np.angle(scatter_field[:,0])

fig, ax = plt.subplots(2,1, sharex=True,subplot_kw={'projection': 'polar'})
ax[0].plot(angles, phases)
ax[0].set_xlabel('angles')
ax[0].set_ylabel(r'phase ($\degree$)')
ax[1].plot(angles, intensity)
ax[1].set_xlabel('angles')
ax[1].set_ylabel('intensity (au)')
plt.show()