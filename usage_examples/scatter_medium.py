import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt


media = n_water + np.array([0, 0.02, 0.04])
scatter_phases = []
scatter_intensities = []
camera_intensities = []
interference_intensities = []

for n in media:
    params = DesignParams(scat_mat="gold", diameter=40, n_medium=n)
    num = 100
    intensity = np.zeros(num)
    scatter_phase = np.zeros(num)
    intensities = np.zeros(num)
    wavelens = np.linspace(450, 650, num)
    for i, wavelen in enumerate(wavelens):
        params.wavelen = wavelen
        scatter_field = calculate_scatter_field(params)
        intensity[i] = np.sum(np.abs(scatter_field))
        scatter_phase[i] = np.degrees(np.angle(scatter_field[0]))

    
    scatter_intensities.append(intensity)
    camera_intensities.append(intensities)
    scatter_phases.append(scatter_phase)


imax = np.argmax(scatter_intensities[0])
print(len(scatter_intensities[0]), len(scatter_phases[0]))
print('plotting')
fig, ax = plt.subplots(1,2, sharex=True)
for intensity, phase, n in zip(scatter_intensities, scatter_phases, media):
    ax[0].plot(wavelens, intensity, label=f'n_medium={n:.2f}')
    ax[1].plot(wavelens, phase, label=f'n_medium={n:.2f}')


fig.suptitle('40nm AuNS plasmon resonances')
ax[0].set_title('scatter amplitude')
ax[0].set_xlabel('wavelength (nm)')
ax[0].set_ylabel('field strength (au)')
ax[0].legend()

ax[1].set_title('scatter phase')
ax[1].set_ylabel(r'phase ($\degree$)')
ax[1].set_xlabel('wavelength (nm)')
plt.show()