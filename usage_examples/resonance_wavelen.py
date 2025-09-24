import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt


diameters = np.array([15, 30, 40, 50, 100])
scatter_phases = []
scatter_intensities = []

for d in diameters:
    params = DesignParams(scat_mat="gold", diameter=d, n_medium=1.33)
    num = 1000
    intensity = np.zeros(num)
    wavelens = np.linspace(450, 650, num)
    for i, wavelen in enumerate(wavelens):
        params.wavelen = wavelen
        scatter_field = calculate_scatter_field(params)
        intensity[i] = np.abs(scatter_field[0])
    
    scatter_intensities.append(intensity)
    # scatter_phases.append(scatter_phase)
    


# print(f'max iscat deriv wavelen: {wavelens[np.argmax(np.diff(center_intensity_if))]}')
# print(f'max scat deriv wavelen: {wavelens[np.argmax(np.diff(center_intensity_scat))]}')
for intensity in scatter_intensities:
    print(f'resonance wavelength:{wavelens[np.argmax(intensity)]}')

# fig, ax = plt.subplots(2,3, sharex=True)
# ax[0,0].plot(wavelens, intensity)
# ax[0,0].set_title('spectrum')
# ax[0,0].set_ylabel('intensity (au)')

# ax[1,0].plot(wavelens, np.rad2deg(scatter_phase))
# ax[1,0].set_title('scatter phase')
# ax[1,0].set_ylabel('phase (degrees)')

# ax[0,1].plot(wavelens, center_intensity_scat)
# ax[0,1].set_title('psf contrast')
# ax[0,1].set_xlabel('wavelength (nm)')
# ax[0,1].set_ylabel('intensity (au)')

# ax[1,1].plot(wavelens, center_intensity_if)
# ax[1,1].set_title('ipsf contrast')
# ax[1,1].set_xlabel('wavelength (nm)')
# ax[1,0].set_ylabel('intensity (au)')

# ax[0,2].plot(wavelens[:-1], np.diff(center_intensity_scat)/np.diff(wavelens))
# ax[0,2].set_title('psf intensity derivative')
# ax[0,2].set_xlabel('wavelength (nm)')
# ax[0,2].set_ylabel(r'$dI/d \lambda $')

# ax[1,2].plot(wavelens[:-1], np.diff(center_intensity_if)/np.diff(wavelens))
# ax[1,2].set_title('ipsf intensity derivative')
# ax[1,2].set_xlabel('wavelength (nm)')
# ax[1,2].set_ylabel(r'$dI/d \lambda $')

fig, ax = plt.subplots(2,1, sharex=True)
for intensity, n in zip(scatter_intensities, diameters):
    ax[0].plot(wavelens, intensity/np.max(intensity), label=f'n={n:.2f}')

# for intensity in scatter_phases:
#     #intensity = intensity/np.max(intensity)
#     ax[1].plot(wavelens, intensity)

# fig.suptitle('AuNS plasmon resonances')
# ax[0].set_xlabel('wavelength(nm)')
# ax[0].legend()
# ax[0].set_ylabel(r'Intensity (au)')
# plt.show()
