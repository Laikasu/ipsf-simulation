import numpy as np


import sys, os
from model import *

import matplotlib.pyplot as plt


params = DesignParams(
    scat_mat="gold",
    diameter=30,
    n_medium=1)


z_focus = np.linspace(-0.1, 0.1, 5)
oil_intensities = []
for z in z_focus:
    params.z_focus = z
    num = 100
    wavelens = np.linspace(400, 600, num)
    intensity = np.zeros(num)
    for i, wavelen in enumerate(wavelens):
        params.wavelen = wavelen
        scatter_field = calculate_scatter_field(params)
        # 1 pixel camera
        params.roi_size = params.pxsize/params.magnification
        camera = Camera(params)
        intensities = calculate_intensities(scatter_field, params, camera, r_resolution=2)
        intensity[i] = intensities['if'][0,0]
    oil_intensities.append(intensity)

fig, ax = plt.subplots(1,1)
for intensity, z in zip(oil_intensities, z_focus):
    ax.plot(wavelens, intensity, label=f'z_focus = {z:.2f} micron')
ax.set_title('Effect of aberrations')
ax.legend()
ax.set_ylabel('contrast')

plt.show()