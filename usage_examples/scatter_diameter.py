import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt

from scipy.constants import *


diameters = np.array([15,40,100])
scatter_phases = []
scatter_intensities = []
camera_intensities = []
interference_intensities = []

for d in diameters:
    params = DesignParams(scat_mat="gold", diameter=d)
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

print('plotting')
fig, ax = plt.subplots(1,2, sharex=True)
for intensity, phase, n in zip(scatter_intensities, scatter_phases, diameters):
    ax[0].plot(wavelens, intensity/np.max(intensity), label=f'diameter={n:.0f}')
    ax[1].plot(wavelens, phase, label=f'diameter={n:.0f}')


fig.suptitle('AuNS plasmon resonances')
ax[0].set_title('scatter amplitude')
ax[0].set_xlabel('wavelength (nm)')
ax[0].set_ylabel('field strength (au)')
ax[0].legend()

ax[1].set_title('scatter phase')
ax[1].set_ylabel(r'phase ($\degree$)')
ax[1].set_xlabel('wavelength (nm)')
plt.show()