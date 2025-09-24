import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt


media = n_water + np.linspace(0, 0.01, 100)

params = DesignParams(scat_mat="gold", diameter=40, wavelen=523.3)

scatter_phases = []
scatter_intensities = []
camera_intensities = []

for n in media:
    params.n_medium = n
    scatter_field = calculate_scatter_field(params)
    scatter_intensities.append(np.sum(np.abs(scatter_field)))
    scatter_phases.append(np.degrees(np.angle(scatter_field[0])))
    # 1 pixel camera
    params.roi_size = params.pxsize/params.magnification
    camera = Camera(params)
    camera_intensities.append(np.squeeze(calculate_intensities(scatter_field, params, camera, 2)['if']))

phase = scatter_phases-scatter_phases[0]
fig, ax = plt.subplots(1,2, sharex=True)
ax[0].plot(media, scatter_intensities/np.min(scatter_intensities), label=f'n_medium={n:.2f}')
ax[1].plot(media, -np.sin(np.radians(phase)), label=f'n_medium={n:.2f}')
#ax[2].plot(media, camera_intensities, label=f'n_medium={n:.2f}')


fig.suptitle('40nm AuNS plasmon resonances')
ax[0].set_title('scatter amplitude')
ax[0].set_xlabel('refractive index')
ax[0].set_ylabel('field strength (au)')
ax[0].legend()

ax[1].set_title('scatter phase')
ax[1].set_ylabel(r'phase ($\degree$)')
ax[1].set_xlabel('refractive index')
plt.show()