import numpy as np


import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from model import *

import matplotlib.pyplot as plt


params = DesignParams(scat_mat="gold", diameter=30)

num = 100
intensity = np.zeros(num)
scatter_phase = np.zeros(num)
center_intensity_if = np.zeros(num)
center_intensity_scat = np.zeros(num)
wavelens = np.linspace(350, 650, num)
for i, wavelen in enumerate(wavelens):
    params.wavelen = wavelen
    scatter_field = calculate_scatter_field(params)
    intensity[i] = np.abs(scatter_field[0,0,0])
    scatter_phase[i] = np.angle(scatter_field[0,0,0])

    # 1 pixel camera
    params.roi_size = params.pxsize/params.magnification
    camera = Camera(params)
    intensities = calculate_intensities(scatter_field, params, camera, r_resolution=2)
    center_intensity_if[i] = intensities['if'][0,0]
    center_intensity_scat[i] = intensities['scat'][0,0]


print(f'max iscat deriv wavelen: {wavelens[np.argmax(np.diff(center_intensity_if))]}')
print(f'max scat deriv wavelen: {wavelens[np.argmax(np.diff(center_intensity_scat))]}')
print(f'resonance wavelength:{wavelens[np.argmax(intensity)]}')

fig, ax = plt.subplots(2,3, sharex=True)
ax[0,0].plot(wavelens, intensity)
ax[0,0].set_title('spectrum')
ax[0,0].set_ylabel('intensity (au)')

ax[1,0].plot(wavelens, np.rad2deg(scatter_phase))
ax[1,0].set_title('scatter phase')
ax[1,0].set_ylabel('phase (degrees)')

ax[0,1].plot(wavelens, center_intensity_scat)
ax[0,1].set_title('psf contrast')
ax[0,1].set_xlabel('wavelength (nm)')
ax[0,1].set_ylabel('intensity (au)')

ax[1,1].plot(wavelens, center_intensity_if)
ax[1,1].set_title('ipsf contrast')
ax[1,1].set_xlabel('wavelength (nm)')
ax[1,0].set_ylabel('intensity (au)')

ax[0,2].plot(wavelens[:-1], np.diff(center_intensity_scat)/np.diff(wavelens))
ax[0,2].set_title('psf intensity derivative')
ax[0,2].set_xlabel('wavelength (nm)')
ax[0,2].set_ylabel(r'$dI/d \lambda $')

ax[1,2].plot(wavelens[:-1], np.diff(center_intensity_if)/np.diff(wavelens))
ax[1,2].set_title('ipsf intensity derivative')
ax[1,2].set_xlabel('wavelength (nm)')
ax[1,2].set_ylabel(r'$dI/d \lambda $')

plt.show()